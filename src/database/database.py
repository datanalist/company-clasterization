import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
import os

# Путь к базе данных
DB_PATH = "database.db"


def get_db_connection():
    """Создает соединение с базой данных"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Чтобы получать результаты в виде словарей
    return conn


def initialize_database():
    """Инициализирует базу данных, создавая необходимые таблицы"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Таблица пользователей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        credits INTEGER NOT NULL DEFAULT 100,
        created_at TEXT NOT NULL
    )
    ''')
    
    # Таблица результатов кластеризации
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS clustering_results (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        status TEXT NOT NULL,
        date_range TEXT NOT NULL,
        clusters TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        visualization_path TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
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


def update_user_credits(user_id: int, new_credits: int) -> bool:
    """Обновляет количество кредитов пользователя"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    UPDATE users
    SET credits = ?
    WHERE id = ?
    ''', (new_credits, user_id))
    
    conn.commit()
    conn.close()
    return True


def save_clustering_result(result: dict) -> bool:
    """Сохраняет результат кластеризации в базу данных"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Сериализуем JSON-поля
    date_range_json = json.dumps(result["date_range"])
    clusters_json = json.dumps(result.get("clusters", None))
    
    cursor.execute('''
    INSERT INTO clustering_results
    (id, user_id, status, date_range, clusters, error, created_at, visualization_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result["id"],
        result["user_id"],
        result["status"],
        date_range_json,
        clusters_json,
        result.get("error", None),
        result["created_at"],
        result.get("visualization_path", None)
    ))
    
    conn.commit()
    conn.close()
    return True


def get_clustering_results(user_id: int) -> List[dict]:
    """Получает все результаты кластеризации для пользователя"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, user_id, status, date_range, clusters, error, created_at, visualization_path
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
        if result_dict["clusters"]:
            result_dict["clusters"] = json.loads(result_dict["clusters"])
        processed_results.append(result_dict)
    
    return processed_results


def get_clustering_result_by_id(result_id: str) -> Optional[dict]:
    """Получает результат кластеризации по ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, user_id, status, date_range, clusters, error, created_at, visualization_path
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
    if result_dict["clusters"]:
        result_dict["clusters"] = json.loads(result_dict["clusters"])
    
    return result_dict 