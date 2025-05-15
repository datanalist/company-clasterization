import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import os
import random
import uuid
import math

# Путь к базе данных
DB_PATH = "database.db"


def get_db_connection():
    """Создает соединение с базой данных"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Чтобы получать результаты в виде словарей
    return conn


def initialize_database():
    """
    Инициализация базы данных (создание таблиц, если они не существуют)
    """
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    # Создание таблицы пользователей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        credits INTEGER DEFAULT 100,
        is_admin BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Создание таблицы результатов кластеризации
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS clustering_results (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        status TEXT NOT NULL,
        date_range TEXT NOT NULL,
        clusters TEXT,
        error TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        visualization_path TEXT,
        ml_config TEXT,
        credits_used INTEGER DEFAULT 0,
        progress REAL DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Создание таблицы ML-моделей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ml_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        description TEXT,
        config TEXT NOT NULL,
        is_default BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Создание таблицы пакетов кредитов
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS credit_packages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        credits INTEGER NOT NULL,
        price REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Создание таблицы покупок кредитов
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS credit_purchases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        package_id INTEGER NOT NULL,
        credits INTEGER NOT NULL,
        price REAL NOT NULL,
        payment_method TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (package_id) REFERENCES credit_packages (id)
    )
    ''')
    
    # Создание таблицы истории кредитов
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS credit_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        amount INTEGER NOT NULL,
        balance_after INTEGER NOT NULL,
        transaction_type TEXT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Создание таблицы математических задач
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS math_problems (
        id TEXT PRIMARY KEY,
        problem TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        solved_by INTEGER,
        solved_at TIMESTAMP,
        FOREIGN KEY (solved_by) REFERENCES users (id)
    )
    ''')
    
    # Создание таблицы новостных файлов
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS news_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        file_path TEXT NOT NULL,
        news_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        news_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'ready',
        metadata TEXT,
        UNIQUE(source, news_date)
    )
    ''')
    
    # Создание индекса для быстрого поиска файлов по дате и источнику
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_news_files_date_source ON news_files (news_date, source)
    ''')
    
    # Вставка дефолтных ML-моделей, если их нет
    add_default_ml_models(cursor)
    
    # Вставка дефолтных пакетов кредитов, если их нет
    add_default_credit_packages(cursor)
    
    conn.commit()
    conn.close()
    
    # Выполняем миграцию для добавления колонки progress, если она не существует
    add_progress_column_if_not_exists()


def add_progress_column_if_not_exists():
    """
    Добавляет колонку progress в таблицу clustering_results, если она не существует
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Проверяем существование колонки progress
        cursor.execute("PRAGMA table_info(clustering_results)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        # Если колонки progress нет, добавляем ее
        if "progress" not in column_names:
            print("Добавление колонки progress в таблицу clustering_results...")
            cursor.execute('''
            ALTER TABLE clustering_results ADD COLUMN progress REAL DEFAULT 0
            ''')
            conn.commit()
            print("Колонка progress успешно добавлена")
        
    except Exception as e:
        print(f"Ошибка при добавлении колонки progress: {e}")
        conn.rollback()
    finally:
        conn.close()


def migrate_database_schema():
    """
    Выполняет миграцию схемы базы данных с model_config на ml_config
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Создаем новую таблицу с правильной схемой
        cursor.execute('''
        CREATE TABLE clustering_results_new (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            date_range TEXT NOT NULL,
            clusters TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            visualization_path TEXT,
            ml_config TEXT,
            credits_used INTEGER DEFAULT 10,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Копируем данные из старой таблицы в новую
        cursor.execute('''
        INSERT INTO clustering_results_new
        SELECT id, user_id, status, date_range, clusters, error, created_at, visualization_path, model_config, credits_used
        FROM clustering_results
        ''')
        
        # Удаляем старую таблицу
        cursor.execute('DROP TABLE clustering_results')
        
        # Переименовываем новую таблицу
        cursor.execute('ALTER TABLE clustering_results_new RENAME TO clustering_results')
        
        conn.commit()
        print("Миграция базы данных успешно завершена")
        
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при миграции базы данных: {e}")
        
    finally:
        conn.close()


def add_user(user_data: dict) -> dict:
    """Добавляет нового пользователя в базу данных"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    user_data["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
    INSERT INTO users (username, email, hashed_password, credits, created_at)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        user_data["username"],
        user_data["email"],
        user_data["hashed_password"],
        user_data["credits"],
        user_data["created_at"]
    ))
    
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    user_data["id"] = user_id
    return user_data


def get_user_by_username(username: str) -> Optional[dict]:
    """Получает пользователя по имени пользователя"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, username, email, hashed_password, credits, created_at
    FROM users
    WHERE username = ?
    ''', (username,))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None


def update_user_credits(user_id: int, new_credits: int, transaction_type: str = "usage", description: str = "") -> bool:
    """Обновляет количество кредитов пользователя и записывает транзакцию"""
    try:
        # Первое соединение - получаем текущие кредиты
        conn1 = get_db_connection()
        cursor1 = conn1.cursor()
        cursor1.execute('SELECT credits FROM users WHERE id = ?', (user_id,))
        current_credits = cursor1.fetchone()[0]
        conn1.close()  # Закрываем первое соединение
        
        # Вычисляем сумму транзакции
        amount = new_credits - current_credits
        
        # Второе соединение - обновляем кредиты
        conn2 = get_db_connection()
        cursor2 = conn2.cursor()
        cursor2.execute('UPDATE users SET credits = ? WHERE id = ?', (new_credits, user_id))
        conn2.commit()
        conn2.close()  # Закрываем второе соединение
        
        # Третье соединение - записываем транзакцию
        conn3 = get_db_connection()
        cursor3 = conn3.cursor()
        cursor3.execute('''
        INSERT INTO credit_transactions (user_id, amount, transaction_type, description, created_at)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            amount,
            transaction_type,
            description,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn3.commit()
        conn3.close()  # Закрываем третье соединение
        
        return True
    except Exception as e:
        print(f"Ошибка при обновлении кредитов: {e}")
        return False


def save_clustering_result(result: dict) -> bool:
    """Сохраняет результат кластеризации в базу данных"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Проверяем, существует ли колонка progress
    cursor.execute("PRAGMA table_info(clustering_results)")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    has_progress_column = "progress" in column_names
    
    # Сериализуем JSON-поля
    date_range_json = json.dumps(result["date_range"])
    clusters_json = json.dumps(result.get("clusters", None))
    ml_config_json = json.dumps(result.get("ml_config", None))
    
    # Формируем запрос в зависимости от наличия колонки progress
    if has_progress_column:
        cursor.execute('''
        INSERT INTO clustering_results
        (id, user_id, status, date_range, clusters, error, created_at, visualization_path, ml_config, credits_used, progress)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result["id"],
            result["user_id"],
            result["status"],
            date_range_json,
            clusters_json,
            result.get("error", None),
            result["created_at"],
            result.get("visualization_path", None),
            ml_config_json,
            result.get("credits_used", 10),
            result.get("progress", 0)
        ))
    else:
        cursor.execute('''
        INSERT INTO clustering_results
        (id, user_id, status, date_range, clusters, error, created_at, visualization_path, ml_config, credits_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result["id"],
            result["user_id"],
            result["status"],
            date_range_json,
            clusters_json,
            result.get("error", None),
            result["created_at"],
            result.get("visualization_path", None),
            ml_config_json,
            result.get("credits_used", 10)
        ))
    
    conn.commit()
    conn.close()
    return True


def get_clustering_results(user_id: int) -> List[dict]:
    """Получает все результаты кластеризации для пользователя"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Проверяем структуру таблицы
    cursor.execute("PRAGMA table_info(clustering_results)")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    ml_config_exists = "ml_config" in column_names
    model_config_exists = "model_config" in column_names
    
    # Формируем SQL-запрос в зависимости от наличия столбцов
    select_fields = [
        "id", "user_id", "status", "date_range", "clusters", "error", 
        "created_at", "visualization_path"
    ]
    
    if ml_config_exists:
        select_fields.append("ml_config")
    elif model_config_exists:
        select_fields.append("model_config as ml_config")
        
    if "credits_used" in column_names:
        select_fields.append("credits_used")
    
    if "progress" in column_names:
        select_fields.append("progress")
    
    fields_str = ", ".join(select_fields)
    
    cursor.execute(f'''
    SELECT {fields_str}
    FROM clustering_results
    WHERE user_id = ?
    ORDER BY created_at DESC
    ''', (user_id,))
    
    results = cursor.fetchall()
    conn.close()
    
    # Десериализуем JSON-поля
    processed_results = []
    for result in results:
        result_dict = dict(result)
        
        result_dict["date_range"] = json.loads(result_dict["date_range"])
        
        if result_dict.get("clusters"):
            result_dict["clusters"] = json.loads(result_dict["clusters"])
            
        if "ml_config" in result_dict and result_dict["ml_config"]:
            result_dict["ml_config"] = json.loads(result_dict["ml_config"])
            
        # Если ml_config отсутствует, добавляем пустой словарь
        if "ml_config" not in result_dict:
            result_dict["ml_config"] = {}
            
        # Если credits_used отсутствует, устанавливаем значение по умолчанию
        if "credits_used" not in result_dict:
            result_dict["credits_used"] = 10
        
        # Если progress отсутствует, устанавливаем значение по умолчанию
        if "progress" not in result_dict:
            result_dict["progress"] = 0
            
        processed_results.append(result_dict)
    
    return processed_results


def get_clustering_result_by_id(result_id: str) -> Optional[dict]:
    """Получает результат кластеризации по ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Проверяем структуру таблицы
    cursor.execute("PRAGMA table_info(clustering_results)")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    ml_config_exists = "ml_config" in column_names
    model_config_exists = "model_config" in column_names
    
    # Формируем SQL-запрос в зависимости от наличия столбцов
    select_fields = [
        "id", "user_id", "status", "date_range", "clusters", "error", 
        "created_at", "visualization_path"
    ]
    
    if ml_config_exists:
        select_fields.append("ml_config")
    elif model_config_exists:
        select_fields.append("model_config as ml_config")
        
    if "credits_used" in column_names:
        select_fields.append("credits_used")
    
    if "progress" in column_names:
        select_fields.append("progress")
    
    fields_str = ", ".join(select_fields)
    
    cursor.execute(f'''
    SELECT {fields_str}
    FROM clustering_results
    WHERE id = ?
    ''', (result_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return None
    
    # Десериализуем JSON-поля
    result_dict = dict(result)
    
    result_dict["date_range"] = json.loads(result_dict["date_range"])
    
    if result_dict.get("clusters"):
        result_dict["clusters"] = json.loads(result_dict["clusters"])
        
    if "ml_config" in result_dict and result_dict["ml_config"]:
        result_dict["ml_config"] = json.loads(result_dict["ml_config"])
        
    # Если ml_config отсутствует, добавляем пустой словарь
    if "ml_config" not in result_dict:
        result_dict["ml_config"] = {}
        
    # Если credits_used отсутствует, устанавливаем значение по умолчанию
    if "credits_used" not in result_dict:
        result_dict["credits_used"] = 10
    
    # Если progress отсутствует, устанавливаем значение по умолчанию
    if "progress" not in result_dict:
        result_dict["progress"] = 0
    
    return result_dict


# Функции для работы с ML-моделями
def get_ml_models(model_type: Optional[str] = None) -> List[dict]:
    """Получает список ML-моделей определенного типа или всех моделей"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if model_type:
        cursor.execute('''
        SELECT id, name, type, config, is_default, created_at
        FROM ml_models
        WHERE type = ?
        ORDER BY is_default DESC, name
        ''', (model_type,))
    else:
        cursor.execute('''
        SELECT id, name, type, config, is_default, created_at
        FROM ml_models
        ORDER BY type, is_default DESC, name
        ''')
    
    models = cursor.fetchall()
    conn.close()
    
    # Десериализуем JSON-поля
    processed_models = []
    for model in models:
        model_dict = dict(model)
        model_dict["config"] = json.loads(model_dict["config"])
        processed_models.append(model_dict)
    
    return processed_models


def get_default_ml_models() -> dict:
    """Получает словарь с дефолтными моделями по каждому типу"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, name, type, config, created_at
    FROM ml_models
    WHERE is_default = 1
    ''')
    
    models = cursor.fetchall()
    conn.close()
    
    default_models = {}
    for model in models:
        model_dict = dict(model)
        model_dict["config"] = json.loads(model_dict["config"])
        default_models[model_dict["type"]] = model_dict
    
    return default_models


def add_ml_model(model_data: dict) -> dict:
    """Добавляет новую ML-модель"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Если модель отмечена как дефолтная, сбрасываем флаг у других моделей этого типа
    if model_data.get("is_default", False):
        cursor.execute('''
        UPDATE ml_models
        SET is_default = 0
        WHERE type = ?
        ''', (model_data["type"],))
    
    cursor.execute('''
    INSERT INTO ml_models (name, type, config, is_default, created_at)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        model_data["name"],
        model_data["type"],
        json.dumps(model_data["config"]),
        1 if model_data.get("is_default", False) else 0,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    
    model_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    model_data["id"] = model_id
    model_data["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return model_data


# Функции для работы с кредитными пакетами
def get_credit_packages() -> List[dict]:
    """Получает список доступных пакетов кредитов"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, name, credits, price, description
    FROM credit_packages
    ORDER BY credits
    ''')
    
    packages = cursor.fetchall()
    conn.close()
    
    return [dict(package) for package in packages]


def get_credit_package(package_id: int) -> Optional[dict]:
    """Получает пакет кредитов по ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, name, credits, price, description
    FROM credit_packages
    WHERE id = ?
    ''', (package_id,))
    
    package = cursor.fetchone()
    conn.close()
    
    if package:
        return dict(package)
    return None


def purchase_credits(user_id: int, package_id: int, payment_method: str) -> bool:
    """Обрабатывает покупку кредитов пользователем"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Получаем информацию о пакете
    cursor.execute('SELECT credits, name FROM credit_packages WHERE id = ?', (package_id,))
    package = cursor.fetchone()
    
    if not package:
        conn.close()
        return False
    
    # Получаем текущие кредиты пользователя
    cursor.execute('SELECT credits FROM users WHERE id = ?', (user_id,))
    current_credits = cursor.fetchone()[0]
    
    # Новое количество кредитов
    new_credits = current_credits + package['credits']
    
    # Обновляем кредиты пользователя
    cursor.execute('''
    UPDATE users
    SET credits = ?
    WHERE id = ?
    ''', (new_credits, user_id))
    
    # Записываем транзакцию
    description = f"Покупка пакета: {package['name']} ({package['credits']} кредитов), метод: {payment_method}"
    cursor.execute('''
    INSERT INTO credit_transactions (user_id, amount, transaction_type, description, created_at)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        user_id,
        package['credits'],
        "purchase",
        description,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    
    conn.commit()
    conn.close()
    return True


# Функции для аналитики и статистики
def get_user_stats(user_id: int) -> dict:
    """Получает статистику использования сервиса пользователем"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Общее количество задач
    cursor.execute('''
    SELECT COUNT(*) as total_tasks
    FROM clustering_results
    WHERE user_id = ?
    ''', (user_id,))
    total_tasks = cursor.fetchone()[0]
    
    # Успешные задачи
    cursor.execute('''
    SELECT COUNT(*) as successful_tasks
    FROM clustering_results
    WHERE user_id = ? AND status = 'completed'
    ''', (user_id,))
    successful_tasks = cursor.fetchone()[0]
    
    # Неудачные задачи
    cursor.execute('''
    SELECT COUNT(*) as failed_tasks
    FROM clustering_results
    WHERE user_id = ? AND status = 'failed'
    ''', (user_id,))
    failed_tasks = cursor.fetchone()[0]
    
    # Потраченные кредиты
    cursor.execute('''
    SELECT SUM(credits_used) as credits_spent
    FROM clustering_results
    WHERE user_id = ? AND status IN ('completed', 'failed')
    ''', (user_id,))
    credits_spent = cursor.fetchone()[0] or 0
    
    # Среднее количество кластеров в задаче
    cursor.execute('''
    SELECT clusters, id
    FROM clustering_results
    WHERE user_id = ? AND status = 'completed'
    ''', (user_id,))
    
    clusters_data = cursor.fetchall()
    
    total_clusters = 0
    for cluster_info in clusters_data:
        if cluster_info['clusters']:
            clusters = json.loads(cluster_info['clusters'])
            total_clusters += len(clusters)
    
    avg_clusters = total_clusters / successful_tasks if successful_tasks > 0 else 0
    
    conn.close()
    
    return {
        "total_clustering_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "credits_spent": credits_spent,
        "avg_clusters_per_task": avg_clusters
    }


def get_system_stats() -> dict:
    """Получает общую статистику использования системы"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Проверяем структуру таблицы
    cursor.execute("PRAGMA table_info(clustering_results)")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    ml_config_exists = "ml_config" in column_names
    model_config_exists = "model_config" in column_names
    credits_used_exists = "credits_used" in column_names
    
    # Общее количество пользователей
    cursor.execute('SELECT COUNT(*) as total_users FROM users')
    total_users = cursor.fetchone()[0]
    
    # Активные пользователи (выполнили хотя бы одну задачу)
    cursor.execute('''
    SELECT COUNT(DISTINCT user_id) as active_users
    FROM clustering_results
    ''')
    active_users = cursor.fetchone()[0]
    
    # Общее количество задач
    cursor.execute('SELECT COUNT(*) as total_tasks FROM clustering_results')
    total_tasks = cursor.fetchone()[0]
    
    # Общее количество потраченных кредитов
    if credits_used_exists:
        cursor.execute('''
        SELECT SUM(credits_used) as credits_spent
        FROM clustering_results
        WHERE status IN ('completed', 'failed')
        ''')
        credits_spent = cursor.fetchone()[0] or 0
    else:
        # Если столбца credits_used нет, предполагаем 10 кредитов на задачу
        cursor.execute('''
        SELECT COUNT(*) * 10 as credits_spent
        FROM clustering_results
        WHERE status IN ('completed', 'failed')
        ''')
        credits_spent = cursor.fetchone()[0] or 0
    
    # Популярные модели
    popular_models = {}
    
    # Определяем, какой столбец использовать для конфигурации моделей
    config_column = "ml_config" if ml_config_exists else "model_config" if model_config_exists else None
    
    if config_column:
        cursor.execute(f'''
        SELECT {config_column}, COUNT(*) as count
        FROM clustering_results
        WHERE status = 'completed' AND {config_column} IS NOT NULL
        GROUP BY {config_column}
        ORDER BY count DESC
        LIMIT 5
        ''')
        
        popular_models_data = cursor.fetchall()
        
        for model_data in popular_models_data:
            config_json = model_data[0]
            count = model_data[1]
            if config_json:
                model_config = json.loads(config_json)
                model_name = f"{model_config.get('embedding', 'Default')}/{model_config.get('reduction', 'Default')}/{model_config.get('clustering', 'Default')}"
                popular_models[model_name] = count
    
    conn.close()
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_tasks": total_tasks,
        "credits_spent": credits_spent,
        "popular_models": popular_models
    }


def get_credit_history(user_id: int) -> List[dict]:
    """Получает историю транзакций кредитов пользователя"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, amount, transaction_type, description, created_at
    FROM credit_transactions
    WHERE user_id = ?
    ORDER BY created_at DESC
    ''', (user_id,))
    
    transactions = cursor.fetchall()
    conn.close()
    
    return [dict(transaction) for transaction in transactions]


def save_math_problem(problem: str) -> dict:
    """Сохраняет математическую задачу в базу данных"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Проверяем наличие таблицы
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS math_problems (
        id TEXT PRIMARY KEY,
        problem TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        used INTEGER DEFAULT 0
    )
    ''')
    
    # Разбираем выражение и вычисляем ответ
    # В целях безопасности используем eval с ограниченным набором операций
    # Выражение должно содержать только числа и операторы +, -, *, /
    sanitized_problem = problem.replace(" ", "")
    for char in sanitized_problem:
        if char not in "0123456789+-*/().":
            raise ValueError("Недопустимые символы в математическом выражении")
    
    # Вычисляем ответ
    answer = eval(sanitized_problem)
    
    # Создаем уникальный ID
    problem_id = str(uuid.uuid4())
    created_at = datetime.now()
    expires_at = created_at + timedelta(minutes=10)  # Задача действительна 10 минут
    
    cursor.execute('''
    INSERT INTO math_problems (id, problem, answer, created_at, expires_at, used)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        problem_id,
        problem,
        answer,
        created_at.strftime("%Y-%m-%d %H:%M:%S"),
        expires_at.strftime("%Y-%m-%d %H:%M:%S"),
        0
    ))
    
    conn.commit()
    conn.close()
    
    return {
        "id": problem_id,
        "problem": problem,
        "expires_at": expires_at.strftime("%Y-%m-%d %H:%M:%S")
    }


def get_math_problem() -> dict:
    """Генерирует случайную математическую задачу"""
    operations = ['+', '-', '*', '/']
    op = random.choice(operations)
    
    # Генерируем числа в зависимости от операции
    if op == '+':
        a = random.randint(1, 100)
        b = random.randint(1, 100)
    elif op == '-':
        a = random.randint(1, 100)
        b = random.randint(1, a)  # Чтобы результат был положительным
    elif op == '*':
        a = random.randint(1, 12)
        b = random.randint(1, 12)
    else:  # op == '/'
        b = random.randint(1, 10)
        a = b * random.randint(1, 10)  # Чтобы результат был целым числом
    
    problem = f"{a} {op} {b}"
    return save_math_problem(problem)


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Получает пользователя по ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, username, email, hashed_password, credits, created_at
    FROM users
    WHERE id = ?
    ''', (user_id,))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None


def validate_math_solution(problem_id: str, user_answer: float, user_id: int) -> Tuple[bool, str]:
    """Проверяет решение математической задачи и пополняет кредиты пользователя"""
    try:
        # Первое соединение - получаем задачу
        conn1 = get_db_connection()
        cursor1 = conn1.cursor()
        cursor1.execute('SELECT problem, answer, expires_at, used FROM math_problems WHERE id = ?', (problem_id,))
        problem_data = cursor1.fetchone()
        
        if not problem_data:
            conn1.close()
            return False, "Задача не найдена"
        
        problem, correct_answer, expires_at, used = problem_data
        
        # Проверяем условия
        if used == 1:
            conn1.close()
            return False, "Эта задача уже была решена"
        
        expires_at_dt = datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S")
        if datetime.now() > expires_at_dt:
            conn1.close()
            return False, "Срок действия задачи истек"
        
        conn1.close()  # Закрываем первое соединение
        
        # Проверяем правильность ответа
        if math.isclose(user_answer, correct_answer, rel_tol=1e-9, abs_tol=1e-9):
            # Второе соединение - отмечаем задачу как использованную
            conn2 = get_db_connection()
            cursor2 = conn2.cursor()
            cursor2.execute('UPDATE math_problems SET used = 1 WHERE id = ?', (problem_id,))
            conn2.commit()
            conn2.close()
            
            # Получаем актуальные данные пользователя
            user_data = get_user_by_id(user_id)
            if not user_data:
                return False, "Пользователь не найден"
                
            # Обновляем кредиты (+100)
            result = update_user_credits(
                user_id,
                user_data["credits"] + 100,
                transaction_type="math_reward",
                description="Пополнение за решение математической задачи"
            )
            
            if result:
                return True, "Правильный ответ! Ваш счет пополнен на 100 кредитов."
            else:
                return False, "Ошибка при обновлении кредитов, хотя ответ был правильным"
        else:
            return False, "Неправильный ответ. Попробуйте еще раз."
            
    except Exception as e:
        print(f"Ошибка в validate_math_solution: {e}")
        return False, f"Произошла ошибка: {str(e)}"


def get_news_file(source, news_date):
    """
    Получает информацию о файле с новостями за указанную дату и из указанного источника
    
    Args:
        source (str): Источник новостей (например, "lenta")
        news_date (str или datetime): Дата новостей в формате YYYY-MM-DD или объект datetime
        
    Returns:
        dict или None: Информация о файле или None, если файл не найден
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Преобразуем date в строку, если это объект datetime
    if hasattr(news_date, 'strftime'):
        news_date = news_date.strftime('%Y-%m-%d')
    
    cursor.execute(
        "SELECT id, source, file_path, news_date, created_at, updated_at, news_count, status, metadata FROM news_files WHERE source = ? AND news_date = ?",
        (source, news_date)
    )
    
    file_data = cursor.fetchone()
    conn.close()
    
    if file_data:
        return {
            "id": file_data[0],
            "source": file_data[1],
            "file_path": file_data[2],
            "news_date": file_data[3],
            "created_at": file_data[4],
            "updated_at": file_data[5],
            "news_count": file_data[6],
            "status": file_data[7],
            "metadata": json.loads(file_data[8]) if file_data[8] else {}
        }
    
    return None


def save_news_file(source, news_date, file_path, news_count, status="ready", metadata=None):
    """
    Сохраняет информацию о файле с новостями в базу данных
    
    Args:
        source (str): Источник новостей (например, "lenta")
        news_date (str или datetime): Дата новостей в формате YYYY-MM-DD или объект datetime
        file_path (str): Путь к файлу с новостями
        news_count (int): Количество новостей в файле
        status (str): Статус файла ("ready", "parsing", "error")
        metadata (dict): Дополнительная информация о файле
        
    Returns:
        bool: True, если файл был успешно сохранен, False в противном случае
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Преобразуем date в строку, если это объект datetime
    if hasattr(news_date, 'strftime'):
        news_date = news_date.strftime('%Y-%m-%d')
    
    # Преобразуем metadata в JSON, если он не None
    metadata_json = json.dumps(metadata) if metadata else None
    
    try:
        # Проверяем, существует ли уже запись для этой даты и источника
        cursor.execute(
            "SELECT id FROM news_files WHERE source = ? AND news_date = ?",
            (source, news_date)
        )
        
        existing_file = cursor.fetchone()
        
        if existing_file:
            # Обновляем существующую запись
            cursor.execute(
                """
                UPDATE news_files 
                SET file_path = ?, updated_at = CURRENT_TIMESTAMP, news_count = ?, status = ?, metadata = ?
                WHERE id = ?
                """,
                (file_path, news_count, status, metadata_json, existing_file[0])
            )
        else:
            # Создаем новую запись
            cursor.execute(
                """
                INSERT INTO news_files (source, news_date, file_path, news_count, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source, news_date, file_path, news_count, status, metadata_json)
            )
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Ошибка при сохранении информации о файле новостей: {e}")
        conn.rollback()
        conn.close()
        return False


def get_news_files_in_range(source, start_date, end_date):
    """
    Получает информацию о файлах с новостями за указанный период из указанного источника
    
    Args:
        source (str): Источник новостей (например, "lenta")
        start_date (str или datetime): Начальная дата в формате YYYY-MM-DD или объект datetime
        end_date (str или datetime): Конечная дата в формате YYYY-MM-DD или объект datetime
        
    Returns:
        list: Список файлов с новостями за указанный период
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Преобразуем даты в строки, если это объекты datetime
    if hasattr(start_date, 'strftime'):
        start_date = start_date.strftime('%Y-%m-%d')
    
    if hasattr(end_date, 'strftime'):
        end_date = end_date.strftime('%Y-%m-%d')
    
    cursor.execute(
        """
        SELECT id, source, file_path, news_date, created_at, updated_at, news_count, status, metadata 
        FROM news_files 
        WHERE source = ? AND news_date BETWEEN ? AND ? AND status = 'ready'
        ORDER BY news_date ASC
        """,
        (source, start_date, end_date)
    )
    
    files_data = cursor.fetchall()
    conn.close()
    
    files = []
    for file_data in files_data:
        files.append({
            "id": file_data[0],
            "source": file_data[1],
            "file_path": file_data[2],
            "news_date": file_data[3],
            "created_at": file_data[4],
            "updated_at": file_data[5],
            "news_count": file_data[6],
            "status": file_data[7],
            "metadata": json.loads(file_data[8]) if file_data[8] else {}
        })
    
    return files


def update_news_file_status(file_id, status, metadata=None):
    """
    Обновляет статус и метаданные файла с новостями
    
    Args:
        file_id (int): ID файла
        status (str): Новый статус файла
        metadata (dict): Новые метаданные файла (опционально)
        
    Returns:
        bool: True, если статус был успешно обновлен, False в противном случае
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if metadata is not None:
            metadata_json = json.dumps(metadata)
            cursor.execute(
                """
                UPDATE news_files 
                SET status = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, metadata_json, file_id)
            )
        else:
            cursor.execute(
                """
                UPDATE news_files 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, file_id)
            )
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Ошибка при обновлении статуса файла новостей: {e}")
        conn.rollback()
        conn.close()
        return False


def add_default_ml_models(cursor):
    """
    Добавляет дефолтные ML-модели в базу данных, если их нет
    
    Args:
        cursor: Курсор базы данных
    """
    # Проверяем, есть ли уже модели в базе
    cursor.execute('SELECT COUNT(*) FROM ml_models')
    if cursor.fetchone()[0] > 0:
        return
    
    # Embedding модели
    cursor.execute('''
    INSERT INTO ml_models (name, type, description, config, is_default, created_at)
    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (
        "LaBSE",
        "embedding",
        "Модель для создания многоязычных эмбеддингов текста",
        json.dumps({
            "method": "sentence_transformer",
            "model_name_or_path": "sentence-transformers/LaBSE",
            "max_length": 128
        }),
        1  # Дефолтная модель
    ))
    
    # Reduction модели
    cursor.execute('''
    INSERT INTO ml_models (name, type, description, config, is_default, created_at)
    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (
        "UMAP",
        "reduction",
        "Алгоритм снижения размерности данных",
        json.dumps({
            "method": "umap",
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.1,
            "metric": "euclidean"
        }),
        1  # Дефолтная модель
    ))
    
    # Clustering модели
    cursor.execute('''
    INSERT INTO ml_models (name, type, description, config, is_default, created_at)
    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (
        "HDBSCAN",
        "clustering",
        "Алгоритм кластеризации на основе плотности",
        json.dumps({
            "model_name": "hdbscan",
            "min_cluster_size": 5,
            "min_samples": 3,
            "cluster_selection_epsilon": 0.5
        }),
        1  # Дефолтная модель
    ))


def add_default_credit_packages(cursor):
    """
    Добавляет дефолтные пакеты кредитов в базу данных, если их нет
    
    Args:
        cursor: Курсор базы данных
    """
    # Проверяем, есть ли уже пакеты в базе
    cursor.execute('SELECT COUNT(*) FROM credit_packages')
    if cursor.fetchone()[0] > 0:
        return
    
    # Добавляем пакеты
    cursor.executemany('''
    INSERT INTO credit_packages (name, credits, price, description, created_at)
    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', [
        ("Стартовый", 100, 5.99, "Базовый пакет для начала работы"),
        ("Стандартный", 500, 19.99, "Оптимальный выбор для регулярного использования"),
        ("Премиум", 1000, 29.99, "Выгодное предложение для активных пользователей"),
        ("Бизнес", 5000, 99.99, "Решение для бизнеса с большим объемом данных")
    ])


def update_clustering_progress(task_id: str, progress: float, status: str = None) -> bool:
    """Обновляет прогресс и, опционально, статус задачи кластеризации
    
    Args:
        task_id (str): ID задачи кластеризации
        progress (float): Прогресс выполнения от 0 до 1
        status (str, optional): Новый статус задачи, если требуется обновить
        
    Returns:
        bool: True, если обновление прошло успешно, False в случае ошибки
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Проверяем, существует ли колонка progress
        cursor.execute("PRAGMA table_info(clustering_results)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        has_progress_column = "progress" in column_names
        
        # Если колонки progress нет, добавляем ее
        if not has_progress_column:
            cursor.execute('''
            ALTER TABLE clustering_results ADD COLUMN progress REAL DEFAULT 0
            ''')
            conn.commit()
        
        if status:
            cursor.execute('''
            UPDATE clustering_results
            SET progress = ?, status = ?
            WHERE id = ?
            ''', (progress, status, task_id))
        else:
            cursor.execute('''
            UPDATE clustering_results
            SET progress = ?
            WHERE id = ?
            ''', (progress, task_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Ошибка при обновлении прогресса задачи: {e}")
        conn.rollback()
        conn.close()
        return False 