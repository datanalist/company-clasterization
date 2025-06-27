import sqlite3
import json
from datetime import datetime, timedelta
import random
import uuid
import math
import operator
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Путь к базе данных
DB_PATH = "database.db"

# Безопасный парсер математических выражений
ALLOWED_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def safe_eval_math(expression: str) -> float:
    """
    Безопасное вычисление простых математических выражений.
    Поддерживает только числа и основные операторы: +, -, *, /
    КРИТИЧНО: НЕ используем eval() из-за уязвимостей безопасности!
    """
    # Удаляем пробелы
    expression = expression.replace(" ", "")

    # Проверяем, что выражение содержит только разрешенные символы
    allowed_chars = set("0123456789+-*/.()")
    if not set(expression).issubset(allowed_chars):
        raise ValueError("Недопустимые символы в математическом выражении")

    # Проверяем на потенциально опасные паттерны
    dangerous_patterns = ["__", "eval", "exec", "import", "open", "file"]
    for pattern in dangerous_patterns:
        if pattern in expression.lower():
            raise ValueError("Обнаружены потенциально опасные паттерны")

    try:
        # Простая реализация парсера математических выражений
        # Поддерживает только базовые операции для безопасности
        import ast

        def safe_eval(node):
            if isinstance(node, ast.Constant):  # Числа
                return node.value
            elif isinstance(node, ast.BinOp):  # Бинарные операции
                left = safe_eval(node.left)
                right = safe_eval(node.right)

                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    if right == 0:
                        raise ValueError("Деление на ноль")
                    return left / right
                else:
                    raise ValueError("Неподдерживаемая операция")
            elif isinstance(node, ast.UnaryOp):  # Унарные операции
                operand = safe_eval(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return operand
                elif isinstance(node.op, ast.USub):
                    return -operand
                else:
                    raise ValueError("Неподдерживаемая унарная операция")
            else:
                raise ValueError("Неподдерживаемый тип выражения")

        # Парсим выражение в AST
        tree = ast.parse(expression, mode="eval")
        result = safe_eval(tree.body)

        # Проверяем, что результат - число
        if not isinstance(result, (int, float)):
            raise ValueError("Результат не является числом")

        return float(result)

    except Exception as e:
        logger.error(
            f"Ошибка при вычислении математического выражения '{expression}': {e}"
        )
        raise ValueError(f"Ошибка при вычислении математического выражения: {e}")


def get_db_connection():
    """Создает соединение с базой данных"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Чтобы получать результаты в виде словарей
    return conn


def initialize_database():
    """Инициализирует базу данных и создает необходимые таблицы, если они не существуют"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Таблица для пользователей
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            credits INTEGER DEFAULT 100,
            created_at TEXT NOT NULL
        )
        """)

        # Таблица для результатов кластеризации
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS clustering_results (
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
        """)

        # Таблица для ML-моделей
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            config TEXT NOT NULL,
            is_default INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """)

        # Таблица для пакетов кредитов
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS credit_packages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            credits INTEGER NOT NULL,
            price REAL NOT NULL,
            description TEXT
        )
        """)

        # Таблица для транзакций кредитов
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS credit_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            amount INTEGER NOT NULL,
            transaction_type TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)

        # Таблица для математических задач
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS math_problems (
            id TEXT PRIMARY KEY,
            problem TEXT NOT NULL,
            answer REAL NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER DEFAULT 0
        )
        """)

        # Добавляем дефолтные пакеты кредитов, если их нет
        cursor.execute("SELECT COUNT(*) FROM credit_packages")
        if cursor.fetchone()[0] == 0:
            cursor.executemany(
                """
            INSERT INTO credit_packages (name, credits, price, description)
            VALUES (?, ?, ?, ?)
            """,
                [
                    ("Стартовый", 100, 5.99, "Базовый пакет для начала работы"),
                    (
                        "Стандартный",
                        500,
                        19.99,
                        "Оптимальный выбор для регулярного использования",
                    ),
                    (
                        "Премиум",
                        1000,
                        29.99,
                        "Выгодное предложение для активных пользователей",
                    ),
                    (
                        "Бизнес",
                        5000,
                        99.99,
                        "Решение для бизнеса с большим объемом данных",
                    ),
                ],
            )

        # Добавляем дефолтные ML-модели, если их нет
        cursor.execute("SELECT COUNT(*) FROM ml_models")
        if cursor.fetchone()[0] == 0:
            # Embedding модели
            cursor.execute(
                """
            INSERT INTO ml_models (name, type, config, is_default, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
                (
                    "rubert-tiny2",
                    "embedding",
                    json.dumps(
                        {
                            "model_name_or_path": "cointegrated/rubert-tiny2",
                            "max_length": 128,
                        }
                    ),
                    1,  # Дефолтная модель
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

            # Reduction модели
            cursor.execute(
                """
            INSERT INTO ml_models (name, type, config, is_default, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
                (
                    "UMAP",
                    "reduction",
                    json.dumps(
                        {
                            "method": "umap",
                            "n_neighbors": 15,
                            "n_components": 5,
                            "min_dist": 0.1,
                            "metric": "euclidean",
                        }
                    ),
                    1,  # Дефолтная модель
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

            # Clustering модели
            cursor.execute(
                """
            INSERT INTO ml_models (name, type, config, is_default, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
                (
                    "HDBSCAN",
                    "clustering",
                    json.dumps(
                        {
                            "method": "hdbscan",
                            "min_cluster_size": 5,
                            "min_samples": 3,
                            "cluster_selection_epsilon": 0.5,
                        }
                    ),
                    1,  # Дефолтная модель
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

        conn.commit()

        # Проверяем необходимость миграции
        migrate_database_schema(conn)

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def migrate_database_schema(conn=None):
    """
    Выполняет миграцию схемы базы данных с model_config на ml_config
    """
    if conn is None:
        conn = get_db_connection()
        should_close = True
    else:
        should_close = False

    cursor = conn.cursor()

    try:
        # Проверяем, есть ли колонка model_config
        cursor.execute("PRAGMA table_info(clustering_results)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]

        if "model_config" in column_names and "ml_config" not in column_names:
            # Выполняем миграцию только если это необходимо
            cursor.execute("ALTER TABLE clustering_results ADD COLUMN ml_config TEXT")
            cursor.execute(
                "UPDATE clustering_results SET ml_config = model_config WHERE model_config IS NOT NULL"
            )
            # Не удаляем старую колонку для обратной совместимости

            conn.commit()
            logger.info("Миграция базы данных успешно завершена")

    except Exception as e:
        conn.rollback()
        logger.error(f"Ошибка при миграции базы данных: {e}")

    finally:
        if should_close:
            conn.close()


def add_user(user_data: dict) -> dict:
    """Добавляет нового пользователя в базу данных"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        user_data["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
        INSERT INTO users (username, email, hashed_password, credits, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                user_data["username"],
                user_data["email"],
                user_data["hashed_password"],
                user_data["credits"],
                user_data["created_at"],
            ),
        )

        user_id = cursor.lastrowid
        conn.commit()

        user_data["id"] = user_id
        logger.info(f"Пользователь {user_data['username']} успешно добавлен")
        return user_data

    except Exception as e:
        conn.rollback()
        logger.error(f"Ошибка при добавлении пользователя: {e}")
        raise e
    finally:
        conn.close()


def get_user_by_username(username: str) -> dict | None:
    """Получает пользователя по имени пользователя"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        SELECT id, username, email, hashed_password, credits, created_at
        FROM users
        WHERE username = ?
        """,
            (username,),
        )

        user = cursor.fetchone()

        if user:
            return dict(user)
        return None

    except Exception as e:
        logger.error(f"Ошибка при получении пользователя: {e}")
        return None
    finally:
        conn.close()


def update_user_credits(
    user_id: int,
    new_credits: int,
    transaction_type: str = "usage",
    description: str = "",
) -> bool:
    """Обновляет количество кредитов пользователя и записывает транзакцию"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Получаем текущие кредиты
        cursor.execute("SELECT credits FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        if not result:
            logger.error(f"Пользователь с ID {user_id} не найден")
            return False

        current_credits = result[0]

        # Вычисляем сумму транзакции
        amount = new_credits - current_credits

        # Обновляем кредиты пользователя
        cursor.execute(
            "UPDATE users SET credits = ? WHERE id = ?", (new_credits, user_id)
        )

        # Записываем транзакцию
        cursor.execute(
            """
        INSERT INTO credit_transactions (user_id, amount, transaction_type, description, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                user_id,
                amount,
                transaction_type,
                description,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

        conn.commit()
        logger.info(
            f"Кредиты пользователя {user_id} обновлены: {current_credits} -> {new_credits}"
        )
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Ошибка при обновлении кредитов: {e}")
        return False
    finally:
        conn.close()


def save_clustering_result(result: dict) -> bool:
    """Сохраняет результат кластеризации в базу данных"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Сериализуем JSON-поля
        date_range_json = json.dumps(result["date_range"])
        clusters_json = json.dumps(result.get("clusters", None))
        ml_config_json = json.dumps(result.get("ml_config", None))

        cursor.execute(
            """
        INSERT INTO clustering_results
        (id, user_id, status, date_range, clusters, error, created_at, visualization_path, ml_config, credits_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
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
            ),
        )

        conn.commit()
        logger.info(
            f"Результат кластеризации {result['id']} сохранен со статусом {result['status']}"
        )
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Ошибка при сохранении результата кластеризации: {e}")
        return False
    finally:
        conn.close()


def get_clustering_results(user_id: int) -> list[dict]:
    """Получает все результаты кластеризации для пользователя"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Проверяем структуру таблицы
        cursor.execute("PRAGMA table_info(clustering_results)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        ml_config_exists = "ml_config" in column_names
        model_config_exists = "model_config" in column_names

        # Формируем SQL-запрос в зависимости от наличия столбцов
        select_fields = [
            "id",
            "user_id",
            "status",
            "date_range",
            "clusters",
            "error",
            "created_at",
            "visualization_path",
        ]

        if ml_config_exists:
            select_fields.append("ml_config")
        elif model_config_exists:
            select_fields.append("model_config as ml_config")

        if "credits_used" in column_names:
            select_fields.append("credits_used")

        fields_str = ", ".join(select_fields)

        cursor.execute(
            f"""
        SELECT {fields_str}
        FROM clustering_results
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
            (user_id,),
        )

        results = cursor.fetchall()

        # Десериализуем JSON-поля
        processed_results = []
        for result in results:
            result_dict = dict(result)

            try:
                result_dict["date_range"] = json.loads(result_dict["date_range"])
            except (json.JSONDecodeError, TypeError):
                result_dict["date_range"] = {}

            if result_dict.get("clusters"):
                try:
                    result_dict["clusters"] = json.loads(result_dict["clusters"])
                except (json.JSONDecodeError, TypeError):
                    result_dict["clusters"] = []

            if "ml_config" in result_dict and result_dict["ml_config"]:
                try:
                    result_dict["ml_config"] = json.loads(result_dict["ml_config"])
                except (json.JSONDecodeError, TypeError):
                    result_dict["ml_config"] = {}

            # Если ml_config отсутствует, добавляем пустой словарь
            if "ml_config" not in result_dict:
                result_dict["ml_config"] = {}

            # Если credits_used отсутствует, устанавливаем значение по умолчанию
            if "credits_used" not in result_dict:
                result_dict["credits_used"] = 10

            processed_results.append(result_dict)

        return processed_results

    except Exception as e:
        logger.error(f"Ошибка при получении результатов кластеризации: {e}")
        return []
    finally:
        conn.close()


def get_clustering_result_by_id(result_id: str) -> dict | None:
    """Получает результат кластеризации по ID"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Проверяем структуру таблицы
        cursor.execute("PRAGMA table_info(clustering_results)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        ml_config_exists = "ml_config" in column_names
        model_config_exists = "model_config" in column_names

        # Формируем SQL-запрос в зависимости от наличия столбцов
        select_fields = [
            "id",
            "user_id",
            "status",
            "date_range",
            "clusters",
            "error",
            "created_at",
            "visualization_path",
        ]

        if ml_config_exists:
            select_fields.append("ml_config")
        elif model_config_exists:
            select_fields.append("model_config as ml_config")

        if "credits_used" in column_names:
            select_fields.append("credits_used")

        fields_str = ", ".join(select_fields)

        cursor.execute(
            f"""
        SELECT {fields_str}
        FROM clustering_results
        WHERE id = ?
        """,
            (result_id,),
        )

        result = cursor.fetchone()

        if not result:
            return None

        # Десериализуем JSON-поля
        result_dict = dict(result)

        try:
            result_dict["date_range"] = json.loads(result_dict["date_range"])
        except (json.JSONDecodeError, TypeError):
            result_dict["date_range"] = {}

        if result_dict.get("clusters"):
            try:
                result_dict["clusters"] = json.loads(result_dict["clusters"])
            except (json.JSONDecodeError, TypeError):
                result_dict["clusters"] = []

        if "ml_config" in result_dict and result_dict["ml_config"]:
            try:
                result_dict["ml_config"] = json.loads(result_dict["ml_config"])
            except (json.JSONDecodeError, TypeError):
                result_dict["ml_config"] = {}

        # Если ml_config отсутствует, добавляем пустой словарь
        if "ml_config" not in result_dict:
            result_dict["ml_config"] = {}

        # Если credits_used отсутствует, устанавливаем значение по умолчанию
        if "credits_used" not in result_dict:
            result_dict["credits_used"] = 10

        return result_dict

    except Exception as e:
        logger.error(f"Ошибка при получении результата кластеризации: {e}")
        return None
    finally:
        conn.close()


# Функции для работы с ML-моделями
def get_ml_models(model_type: str | None = None) -> list[dict]:
    """Получает список ML-моделей определенного типа или всех моделей"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        if model_type:
            cursor.execute(
                """
            SELECT id, name, type, config, is_default, created_at
            FROM ml_models
            WHERE type = ?
            ORDER BY is_default DESC, name
            """,
                (model_type,),
            )
        else:
            cursor.execute("""
            SELECT id, name, type, config, is_default, created_at
            FROM ml_models
            ORDER BY type, is_default DESC, name
            """)

        models = cursor.fetchall()

        # Десериализуем JSON-поля
        processed_models = []
        for model in models:
            model_dict = dict(model)
            try:
                model_dict["config"] = json.loads(model_dict["config"])
            except (json.JSONDecodeError, TypeError):
                model_dict["config"] = {}
            processed_models.append(model_dict)

        return processed_models

    except Exception as e:
        logger.error(f"Ошибка при получении ML-моделей: {e}")
        return []
    finally:
        conn.close()


def get_default_ml_models() -> dict:
    """Получает словарь с дефолтными моделями по каждому типу"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT id, name, type, config, created_at
        FROM ml_models
        WHERE is_default = 1
        """)

        models = cursor.fetchall()

        default_models = {}
        for model in models:
            model_dict = dict(model)
            try:
                model_dict["config"] = json.loads(model_dict["config"])
            except (json.JSONDecodeError, TypeError):
                model_dict["config"] = {}
            default_models[model_dict["type"]] = model_dict

        return default_models

    except Exception as e:
        logger.error(f"Ошибка при получении дефолтных моделей: {e}")
        return {}
    finally:
        conn.close()


def add_ml_model(model_data: dict) -> dict:
    """Добавляет новую ML-модель"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Если модель отмечена как дефолтная, сбрасываем флаг у других моделей этого типа
        if model_data.get("is_default", False):
            cursor.execute(
                """
            UPDATE ml_models
            SET is_default = 0
            WHERE type = ?
            """,
                (model_data["type"],),
            )

        cursor.execute(
            """
        INSERT INTO ml_models (name, type, config, is_default, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                model_data["name"],
                model_data["type"],
                json.dumps(model_data["config"]),
                1 if model_data.get("is_default", False) else 0,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

        model_id = cursor.lastrowid
        conn.commit()

        model_data["id"] = model_id
        model_data["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"ML-модель {model_data['name']} типа {model_data['type']} добавлена"
        )
        return model_data

    except Exception as e:
        conn.rollback()
        logger.error(f"Ошибка при добавлении ML-модели: {e}")
        raise e
    finally:
        conn.close()


# Функции для работы с кредитными пакетами
def get_credit_packages() -> list[dict]:
    """Получает список доступных пакетов кредитов"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT id, name, credits, price, description
        FROM credit_packages
        ORDER BY credits
        """)

        packages = cursor.fetchall()
        return [dict(package) for package in packages]

    except Exception as e:
        logger.error(f"Ошибка при получении пакетов кредитов: {e}")
        return []
    finally:
        conn.close()


def get_credit_package(package_id: int) -> dict | None:
    """Получает пакет кредитов по ID"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        SELECT id, name, credits, price, description
        FROM credit_packages
        WHERE id = ?
        """,
            (package_id,),
        )

        package = cursor.fetchone()

        if package:
            return dict(package)
        return None

    except Exception as e:
        logger.error(f"Ошибка при получении пакета кредитов: {e}")
        return None
    finally:
        conn.close()


def purchase_credits(user_id: int, package_id: int, payment_method: str) -> bool:
    """Обрабатывает покупку кредитов пользователем"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Получаем информацию о пакете
        cursor.execute(
            "SELECT credits, name FROM credit_packages WHERE id = ?", (package_id,)
        )
        package = cursor.fetchone()

        if not package:
            logger.error(f"Пакет кредитов с ID {package_id} не найден")
            return False

        # Получаем текущие кредиты пользователя
        cursor.execute("SELECT credits FROM users WHERE id = ?", (user_id,))
        user_result = cursor.fetchone()
        if not user_result:
            logger.error(f"Пользователь с ID {user_id} не найден")
            return False

        current_credits = user_result[0]

        # Новое количество кредитов
        new_credits = current_credits + package["credits"]

        # Обновляем кредиты пользователя
        cursor.execute(
            """
        UPDATE users
        SET credits = ?
        WHERE id = ?
        """,
            (new_credits, user_id),
        )

        # Записываем транзакцию
        description = f"Покупка пакета: {package['name']} ({package['credits']} кредитов), метод: {payment_method}"
        cursor.execute(
            """
        INSERT INTO credit_transactions (user_id, amount, transaction_type, description, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                user_id,
                package["credits"],
                "purchase",
                description,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

        conn.commit()
        logger.info(
            f"Пользователь {user_id} купил пакет {package['name']} за {payment_method}"
        )
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Ошибка при покупке кредитов: {e}")
        return False
    finally:
        conn.close()


# Функции для аналитики и статистики
def get_user_stats(user_id: int) -> dict:
    """Получает статистику использования сервиса пользователем"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Общее количество задач
        cursor.execute(
            """
        SELECT COUNT(*) as total_tasks
        FROM clustering_results
        WHERE user_id = ?
        """,
            (user_id,),
        )
        total_tasks = cursor.fetchone()[0]

        # Успешные задачи
        cursor.execute(
            """
        SELECT COUNT(*) as successful_tasks
        FROM clustering_results
        WHERE user_id = ? AND status = 'completed'
        """,
            (user_id,),
        )
        successful_tasks = cursor.fetchone()[0]

        # Неудачные задачи
        cursor.execute(
            """
        SELECT COUNT(*) as failed_tasks
        FROM clustering_results
        WHERE user_id = ? AND status = 'failed'
        """,
            (user_id,),
        )
        failed_tasks = cursor.fetchone()[0]

        # Потраченные кредиты
        cursor.execute(
            """
        SELECT SUM(credits_used) as credits_spent
        FROM clustering_results
        WHERE user_id = ? AND status IN ('completed', 'failed')
        """,
            (user_id,),
        )
        credits_spent = cursor.fetchone()[0] or 0

        # Среднее количество кластеров в задаче
        cursor.execute(
            """
        SELECT clusters, id
        FROM clustering_results
        WHERE user_id = ? AND status = 'completed'
        """,
            (user_id,),
        )

        clusters_data = cursor.fetchall()

        total_clusters = 0
        for cluster_info in clusters_data:
            if cluster_info["clusters"]:
                try:
                    clusters = json.loads(cluster_info["clusters"])
                    total_clusters += len(clusters)
                except (json.JSONDecodeError, TypeError):
                    pass

        avg_clusters = total_clusters / successful_tasks if successful_tasks > 0 else 0

        return {
            "total_clustering_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "credits_spent": credits_spent,
            "avg_clusters_per_task": avg_clusters,
        }

    except Exception as e:
        logger.error(f"Ошибка при получении статистики пользователя: {e}")
        return {
            "total_clustering_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "credits_spent": 0,
            "avg_clusters_per_task": 0,
        }
    finally:
        conn.close()


def get_system_stats() -> dict:
    """Получает общую статистику использования системы"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Проверяем структуру таблицы
        cursor.execute("PRAGMA table_info(clustering_results)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        ml_config_exists = "ml_config" in column_names
        model_config_exists = "model_config" in column_names
        credits_used_exists = "credits_used" in column_names

        # Общее количество пользователей
        cursor.execute("SELECT COUNT(*) as total_users FROM users")
        total_users = cursor.fetchone()[0]

        # Активные пользователи (выполнили хотя бы одну задачу)
        cursor.execute("""
        SELECT COUNT(DISTINCT user_id) as active_users
        FROM clustering_results
        """)
        active_users = cursor.fetchone()[0]

        # Общее количество задач
        cursor.execute("SELECT COUNT(*) as total_tasks FROM clustering_results")
        total_tasks = cursor.fetchone()[0]

        # Общее количество потраченных кредитов
        if credits_used_exists:
            cursor.execute("""
            SELECT SUM(credits_used) as credits_spent
            FROM clustering_results
            WHERE status IN ('completed', 'failed')
            """)
            credits_spent = cursor.fetchone()[0] or 0
        else:
            # Если столбца credits_used нет, предполагаем 10 кредитов на задачу
            cursor.execute("""
            SELECT COUNT(*) * 10 as credits_spent
            FROM clustering_results
            WHERE status IN ('completed', 'failed')
            """)
            credits_spent = cursor.fetchone()[0] or 0

        # Популярные модели
        popular_models = {}

        # Определяем, какой столбец использовать для конфигурации моделей
        config_column = (
            "ml_config"
            if ml_config_exists
            else "model_config"
            if model_config_exists
            else None
        )

        if config_column:
            cursor.execute(f"""
            SELECT {config_column}, COUNT(*) as count
            FROM clustering_results
            WHERE status = 'completed' AND {config_column} IS NOT NULL
            GROUP BY {config_column}
            ORDER BY count DESC
            LIMIT 5
            """)

            popular_models_data = cursor.fetchall()

            for model_data in popular_models_data:
                config_json = model_data[0]
                count = model_data[1]
                if config_json:
                    try:
                        model_config = json.loads(config_json)
                        model_name = f"{model_config.get('embedding', 'Default')}/{model_config.get('reduction', 'Default')}/{model_config.get('clustering', 'Default')}"
                        popular_models[model_name] = count
                    except (json.JSONDecodeError, TypeError):
                        pass

        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_tasks": total_tasks,
            "credits_spent": credits_spent,
            "popular_models": popular_models,
        }

    except Exception as e:
        logger.error(f"Ошибка при получении системной статистики: {e}")
        return {
            "total_users": 0,
            "active_users": 0,
            "total_tasks": 0,
            "credits_spent": 0,
            "popular_models": {},
        }
    finally:
        conn.close()


def get_credit_history(user_id: int) -> list[dict]:
    """Получает историю транзакций кредитов пользователя"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        SELECT id, amount, transaction_type, description, created_at
        FROM credit_transactions
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
            (user_id,),
        )

        transactions = cursor.fetchall()
        return [dict(transaction) for transaction in transactions]

    except Exception as e:
        logger.error(f"Ошибка при получении истории кредитов: {e}")
        return []
    finally:
        conn.close()


def save_math_problem(problem: str) -> dict:
    """Сохраняет математическую задачу в базу данных"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Проверяем наличие таблицы
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS math_problems (
            id TEXT PRIMARY KEY,
            problem TEXT NOT NULL,
            answer REAL NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER DEFAULT 0
        )
        """)

        # Безопасно вычисляем ответ
        answer = safe_eval_math(problem)

        # Создаем уникальный ID
        problem_id = str(uuid.uuid4())
        created_at = datetime.now()
        expires_at = created_at + timedelta(minutes=10)  # Задача действительна 10 минут

        cursor.execute(
            """
        INSERT INTO math_problems (id, problem, answer, created_at, expires_at, used)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                problem_id,
                problem,
                answer,
                created_at.strftime("%Y-%m-%d %H:%M:%S"),
                expires_at.strftime("%Y-%m-%d %H:%M:%S"),
                0,
            ),
        )

        conn.commit()

        return {
            "id": problem_id,
            "problem": problem,
            "expires_at": expires_at.strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        conn.rollback()
        logger.error(f"Ошибка при сохранении математической задачи: {e}")
        raise e
    finally:
        conn.close()


def get_math_problem() -> dict:
    """Генерирует случайную математическую задачу"""
    operations = ["+", "-", "*", "/"]
    op = random.choice(operations)

    # Генерируем числа в зависимости от операции
    if op == "+":
        a = random.randint(1, 100)
        b = random.randint(1, 100)
    elif op == "-":
        a = random.randint(1, 100)
        b = random.randint(1, a)  # Чтобы результат был положительным
    elif op == "*":
        a = random.randint(1, 12)
        b = random.randint(1, 12)
    else:  # op == '/'
        b = random.randint(1, 10)
        a = b * random.randint(1, 10)  # Чтобы результат был целым числом

    problem = f"{a} {op} {b}"
    return save_math_problem(problem)


def get_user_by_id(user_id: int) -> dict | None:
    """Получает пользователя по ID"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        SELECT id, username, email, hashed_password, credits, created_at
        FROM users
        WHERE id = ?
        """,
            (user_id,),
        )

        user = cursor.fetchone()

        if user:
            return dict(user)
        return None

    except Exception as e:
        logger.error(f"Ошибка при получении пользователя по ID: {e}")
        return None
    finally:
        conn.close()


def validate_math_solution(
    problem_id: str, user_answer: float, user_id: int
) -> tuple[bool, str]:
    """Проверяет решение математической задачи и пополняет кредиты пользователя"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Получаем задачу
        cursor.execute(
            "SELECT problem, answer, expires_at, used FROM math_problems WHERE id = ?",
            (problem_id,),
        )
        problem_data = cursor.fetchone()

        if not problem_data:
            return False, "Задача не найдена"

        problem, correct_answer, expires_at, used = problem_data

        # Проверяем условия
        if used == 1:
            return False, "Эта задача уже была решена"

        expires_at_dt = datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S")
        if datetime.now() > expires_at_dt:
            return False, "Срок действия задачи истек"

        # Проверяем правильность ответа
        if math.isclose(user_answer, correct_answer, rel_tol=1e-9, abs_tol=1e-9):
            # Отмечаем задачу как использованную
            cursor.execute(
                "UPDATE math_problems SET used = 1 WHERE id = ?", (problem_id,)
            )

            # Получаем актуальные данные пользователя
            user_data = get_user_by_id(user_id)
            if not user_data:
                conn.rollback()
                return False, "Пользователь не найден"

            # Обновляем кредиты (+100)
            result = update_user_credits(
                user_id,
                user_data["credits"] + 100,
                transaction_type="math_reward",
                description="Пополнение за решение математической задачи",
            )

            conn.commit()

            if result:
                logger.info(
                    f"Пользователь {user_id} правильно решил задачу {problem_id} и получил 100 кредитов"
                )
                return True, "Правильный ответ! Ваш счет пополнен на 100 кредитов."
            else:
                return (
                    False,
                    "Ошибка при обновлении кредитов, хотя ответ был правильным",
                )
        else:
            logger.info(
                f"Пользователь {user_id} дал неправильный ответ для задачи {problem_id}"
            )
            return False, "Неправильный ответ. Попробуйте еще раз."

    except Exception as e:
        conn.rollback()
        logger.error(f"Ошибка в validate_math_solution: {e}")
        return False, f"Произошла ошибка: {str(e)}"
    finally:
        conn.close()
