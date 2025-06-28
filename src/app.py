from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from datetime import datetime
import pandas as pd
import uuid
import asyncio
import logging

from ml_core.models import (
    EmbeddingModel,
    ReductionModel,
    ClusteringModel,
    create_topic_names,
)
from data.prepare import clean_text
from models.models import (
    ClusteringRequest,
    ClusteringResult,
    User,
    UserCreate,
    MLModelConfig,
    MLModelUpload,
    MLModelsList,
    CreditPackage,
    CreditPurchase,
    UserStats,
    SystemStats,
    MathProblem,
    MathSolution,
)
from database.database import (
    initialize_database,
    add_user,
    get_user_by_username,
    update_user_credits,
    save_clustering_result,
    get_clustering_results,
    get_clustering_result_by_id,
    get_ml_models,
    get_default_ml_models,
    add_ml_model,
    get_credit_packages,
    get_credit_package,
    purchase_credits,
    get_user_stats,
    get_system_stats,
    get_credit_history,
    get_math_problem,
    validate_math_solution,
    get_user_by_id,
    safe_json_dump,
)
from utils.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_admin_user,
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем директории если их нет
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Инициализируем базу данных
try:
    initialize_database()
    logger.info("База данных успешно инициализирована")
except Exception as e:
    logger.error(f"Ошибка при инициализации базы данных: {e}")
    raise

app = FastAPI(title="Кластеризация компаний по новостям")

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "API для кластеризации компаний по новостным данным"}


@app.post("/api/register", response_model=User)
def register_user(user: UserCreate):
    try:
        db_user = get_user_by_username(user.username)
        if db_user:
            raise HTTPException(
                status_code=400, detail="Пользователь уже зарегистрирован"
            )

        hashed_password = get_password_hash(user.password)
        user_data = user.dict()
        user_data.pop("password")
        user_data["hashed_password"] = hashed_password
        user_data["credits"] = 100  # Начальное количество кредитов

        return add_user(user_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при регистрации пользователя: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/api/token")
def login_for_access_token(username: str, password: str):
    try:
        user = get_user_by_username(username)
        if not user or not verify_password(password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверное имя пользователя или пароль",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token = create_access_token(data={"sub": user["username"]})
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при аутентификации: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/users/me", response_model=User)
def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user


@app.post("/api/clustering/start")
async def start_clustering(
    request: ClusteringRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    try:
        # Определяем стоимость в кредитах (можно сделать динамически в зависимости от выбранных моделей)
        credits_cost = 10

        # Проверяем кредиты пользователя
        if current_user["credits"] < credits_cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Недостаточно кредитов",
            )

        # Создаем уникальный идентификатор для задачи
        task_id = str(uuid.uuid4())

        # Запускаем задачу кластеризации в фоновом режиме
        background_tasks.add_task(
            perform_clustering_task,
            task_id=task_id,
            request=request,
            user_id=current_user["id"],
            credits_cost=credits_cost,  # Передаем стоимость задачи
        )

        return {"task_id": task_id, "status": "started"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при запуске кластеризации: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/clustering/results", response_model=list[ClusteringResult])
def get_user_clustering_results(
    response: Response,
    current_user: dict = Depends(get_current_user),
    _t: str = None,  # Параметр для обхода кэширования
):
    try:
        # Добавляем заголовки против кэширования
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        logger.info(
            f"Получение результатов кластеризации для пользователя {current_user['id']}"
        )
        results = get_clustering_results(user_id=current_user["id"])
        logger.info(
            f"Найдено {len(results)} результатов для пользователя {current_user['id']}"
        )
        return results
    except Exception as e:
        logger.error(f"Ошибка при получении результатов кластеризации: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/clustering/results/{task_id}", response_model=ClusteringResult)
def get_clustering_result(task_id: str, current_user: dict = Depends(get_current_user)):
    try:
        result = get_clustering_result_by_id(task_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Результат не найден"
            )

        if result["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Нет доступа к этому результату",
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении результата кластеризации: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/ml/models", response_model=MLModelsList)
def get_available_ml_models(current_user: dict = Depends(get_current_user)):
    """Получает список доступных ML-моделей по типам"""
    try:
        all_models = get_ml_models()

        # Группируем модели по типу
        embedding_models = [
            model for model in all_models if model["type"] == "embedding"
        ]
        reduction_models = [
            model for model in all_models if model["type"] == "reduction"
        ]
        clustering_models = [
            model for model in all_models if model["type"] == "clustering"
        ]

        return {
            "embedding_models": embedding_models,
            "reduction_models": reduction_models,
            "clustering_models": clustering_models,
        }
    except Exception as e:
        logger.error(f"Ошибка при получении ML-моделей: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/ml/models/default")
def get_default_models(current_user: dict = Depends(get_current_user)):
    """Получает набор дефолтных ML-моделей"""
    try:
        default_models = get_default_ml_models()
        return default_models
    except Exception as e:
        logger.error(f"Ошибка при получении дефолтных моделей: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/api/ml/models", response_model=MLModelConfig)
def upload_ml_model(
    model: MLModelUpload, current_user: dict = Depends(get_current_admin_user)
):
    """Добавляет новую ML-модель (требуются права администратора)"""
    try:
        return add_ml_model(model.dict())
    except Exception as e:
        logger.error(f"Ошибка при добавлении ML-модели: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/credits/packages", response_model=list[CreditPackage])
def get_available_credit_packages(current_user: dict = Depends(get_current_user)):
    """Получает список доступных пакетов кредитов"""
    try:
        return get_credit_packages()
    except Exception as e:
        logger.error(f"Ошибка при получении пакетов кредитов: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/api/credits/purchase")
def buy_credits(
    purchase: CreditPurchase, current_user: dict = Depends(get_current_user)
):
    """Покупка кредитов"""
    try:
        # Проверяем существование пакета
        package = get_credit_package(purchase.package_id)
        if not package:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Пакет кредитов не найден"
            )

        # В реальном приложении здесь был бы код для обработки платежа
        # Например, вызов API платежной системы

        # Пока просто добавляем кредиты
        success = purchase_credits(
            current_user["id"], purchase.package_id, purchase.payment_method
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ошибка при покупке кредитов",
            )

        # Получаем обновленную информацию о пользователе
        updated_user = get_user_by_username(current_user["username"])

        return {
            "status": "success",
            "message": f"Кредиты успешно добавлены. Ваш баланс: {updated_user['credits']}",
            "credits": updated_user["credits"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при покупке кредитов: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/credits/history")
def get_user_credit_history(current_user: dict = Depends(get_current_user)):
    """Получает историю транзакций по кредитам пользователя"""
    try:
        return get_credit_history(current_user["id"])
    except Exception as e:
        logger.error(f"Ошибка при получении истории кредитов: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/credits/math-problem", response_model=MathProblem)
def generate_math_problem(current_user: dict = Depends(get_current_user)):
    """Генерирует случайную математическую задачу для пополнения кредитов"""
    try:
        return get_math_problem()
    except Exception as e:
        logger.error(f"Ошибка при генерации математической задачи: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/api/credits/math-solution")
def check_math_solution(
    solution: MathSolution, current_user: dict = Depends(get_current_user)
):
    """Проверяет решение математической задачи и пополняет кредиты при правильном ответе"""
    try:
        success, message = validate_math_solution(
            solution.problem_id, solution.answer, current_user["id"]
        )

        if success:
            # Получаем обновленную информацию о пользователе
            updated_user = get_user_by_username(current_user["username"])

            return {
                "success": True,
                "message": message,
                "credits": updated_user["credits"],
            }
        else:
            return {"success": False, "message": message}
    except Exception as e:
        logger.error(f"Ошибка при проверке математического решения: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/stats/user", response_model=UserStats)
def get_current_user_stats(current_user: dict = Depends(get_current_user)):
    """Получает статистику использования сервиса текущим пользователем"""
    try:
        return get_user_stats(current_user["id"])
    except Exception as e:
        logger.error(f"Ошибка при получении статистики пользователя: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/api/stats/system", response_model=SystemStats)
def get_general_system_stats(current_user: dict = Depends(get_current_admin_user)):
    """Получает общую статистику использования системы (требуются права администратора)"""
    try:
        return get_system_stats()
    except Exception as e:
        logger.error(f"Ошибка при получении системной статистики: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


async def perform_clustering_task(
    task_id: str, request: ClusteringRequest, user_id: int, credits_cost: int
):
    """Выполняет задачу кластеризации в фоновом режиме"""
    try:
        # Импортируем модули для парсинга с обработкой ошибок
        try:
            from src.data.parse_lenta import parse_lenta_news
        except ImportError:
            try:
                from data.parse_lenta import parse_lenta_news
            except ImportError:
                logger.error("Не удалось импортировать модуль parse_lenta_news")
                save_clustering_result(
                    {
                        "id": task_id,
                        "user_id": user_id,
                        "status": "failed",
                        "date_range": {
                            "start_date": request.date_range.start_date,
                            "end_date": request.date_range.end_date,
                        },
                        "error": "Модуль парсинга новостей недоступен",
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "ml_config": request.ml_config or {},
                        "credits_used": 0,
                    }
                )
                return

        # Telegram парсер (в разработке, не используется по умолчанию)
        # from data.parse_tg import parse_forbes_news

        logger.info(f"Начало выполнения задачи кластеризации {task_id}")

        # Определяем период времени для анализа
        start_date = datetime.strptime(request.date_range.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.date_range.end_date, "%Y-%m-%d")

        # Получаем пользовательскую конфигурацию моделей или дефолтные
        ml_config = request.ml_config or {}
        default_models = get_default_ml_models()

        # Определяем, какие модели использовать
        embedding_model_config = ml_config.get(
            "embedding", default_models.get("embedding", {}).get("config", {})
        )
        reduction_model_config = ml_config.get(
            "reduction", default_models.get("reduction", {}).get("config", {})
        )
        clustering_model_config = ml_config.get(
            "clustering", default_models.get("clustering", {}).get("config", {})
        )

        # Получаем данные только из Lenta.ru
        logger.info(
            f"Загрузка новостей с Lenta.ru за период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}"
        )

        try:
            lenta_news_future = asyncio.create_task(
                asyncio.to_thread(
                    lambda: parse_lenta_news(start_date=start_date, end_date=end_date)
                )
            )

            # Ждем выполнения задачи с таймаутом
            lenta_df = await asyncio.wait_for(
                lenta_news_future, timeout=300
            )  # 5 минут таймаут
        except asyncio.TimeoutError:
            error_message = "Таймаут при загрузке новостей"
            logger.error(f"Task {task_id}: {error_message}")
            save_clustering_result(
                {
                    "id": task_id,
                    "user_id": user_id,
                    "status": "failed",
                    "date_range": {
                        "start_date": request.date_range.start_date,
                        "end_date": request.date_range.end_date,
                    },
                    "error": error_message,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ml_config": ml_config,
                    "credits_used": 0,
                }
            )
            return
        except Exception as e:
            error_message = f"Ошибка при загрузке новостей: {str(e)}"
            logger.error(f"Task {task_id}: {error_message}")
            save_clustering_result(
                {
                    "id": task_id,
                    "user_id": user_id,
                    "status": "failed",
                    "date_range": {
                        "start_date": request.date_range.start_date,
                        "end_date": request.date_range.end_date,
                    },
                    "error": error_message,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ml_config": ml_config,
                    "credits_used": 0,
                }
            )
            return

        # Проверка и подготовка lenta_df
        expected_cols_parser = [
            "title",
            "text",
            "date",
            "url",
        ]  # Ожидаемые колонки от парсера
        if lenta_df is None or not isinstance(lenta_df, pd.DataFrame):
            logger.warning(
                f"Task {task_id}: Парсер Lenta.ru вернул None или не DataFrame. Создаем пустой DataFrame."
            )
            lenta_df = pd.DataFrame(columns=expected_cols_parser)

        for col in expected_cols_parser:
            if col not in lenta_df.columns:
                logger.warning(
                    f"Task {task_id}: В DataFrame от Lenta.ru отсутствует колонка: {col}. Добавляем пустой столбец."
                )
                lenta_df[col] = pd.NA

        # Создаем пустой DataFrame для Forbes (Telegram) - в разработке
        forbes_df = pd.DataFrame(columns=lenta_df.columns.tolist())
        forbes_df["source"] = "forbes (в разработке)"

        # Подготавливаем данные Lenta.ru
        lenta_df["source"] = "lenta"

        # Объединение данных (включаем только Lenta.ru)
        news_df = lenta_df.copy()

        # Фильтруем по дате
        final_expected_cols = expected_cols_parser + ["source", "cleaned_text"]

        if (
            not news_df.empty
            and "date" in news_df.columns
            and not news_df["date"].isnull().all()
        ):
            try:
                # Убедимся, что start_date и end_date являются объектами date для сравнения с .dt.date
                s_date = (
                    start_date.date()
                    if isinstance(start_date, datetime)
                    else start_date
                )
                e_date = end_date.date() if isinstance(end_date, datetime) else end_date

                news_df["date"] = pd.to_datetime(news_df["date"])
                news_df = news_df[
                    (news_df["date"].dt.date >= s_date)
                    & (news_df["date"].dt.date <= e_date)
                ]
            except Exception as e_filter:
                logger.error(
                    f"Task {task_id}: Ошибка при конвертации или фильтрации дат в news_df: {e_filter}. Создаем пустой DataFrame."
                )
                news_df = pd.DataFrame(columns=final_expected_cols)
        elif news_df.empty:
            logger.warning(
                f"Task {task_id}: news_df пуст перед фильтрацией по дате. Убеждаемся, что он имеет правильные колонки."
            )
            news_df = pd.DataFrame(columns=final_expected_cols)
        else:
            logger.warning(
                f"Task {task_id}: Колонка 'date' отсутствует в news_df или полностью NaN перед фильтрацией. Создаем пустой DataFrame."
            )
            news_df = pd.DataFrame(columns=final_expected_cols)

        # Проверка на наличие данных после всех фильтраций
        if news_df.empty:
            error_message = f"Не найдено новостей в указанный период ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}) после парсинга и всех фильтраций."
            logger.warning(f"Task {task_id}: {error_message}")
            save_clustering_result(
                {
                    "id": task_id,
                    "user_id": user_id,
                    "status": "failed",
                    "date_range": {
                        "start_date": request.date_range.start_date,
                        "end_date": request.date_range.end_date,
                    },
                    "error": error_message,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ml_config": ml_config,
                    "credits_used": 0,
                }
            )
            return

        logger.info(
            f"Task {task_id}: Загружено {len(news_df)} новостей после начальной фильтрации."
        )

        # Очистка текста
        if "text" in news_df.columns:
            news_df["cleaned_text"] = news_df["text"].apply(clean_text)
        else:
            logger.warning(
                f"Task {task_id}: Колонка 'text' отсутствует для очистки. Пропускаем."
            )
            news_df["cleaned_text"] = ""

        # Загрузка модели эмбеддингов с использованием пользовательской конфигурации
        if "method" not in embedding_model_config:
            embedding_model_config["method"] = "sentence_transformer"

        try:
            embedding_model = EmbeddingModel(**embedding_model_config)
            logger.info(f"Task {task_id}: Модель эмбеддингов загружена")
        except Exception as e:
            error_message = f"Ошибка при загрузке модели эмбеддингов: {str(e)}"
            logger.error(f"Task {task_id}: {error_message}")
            save_clustering_result(
                {
                    "id": task_id,
                    "user_id": user_id,
                    "status": "failed",
                    "date_range": {
                        "start_date": request.date_range.start_date,
                        "end_date": request.date_range.end_date,
                    },
                    "error": error_message,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ml_config": ml_config,
                    "credits_used": 0,
                }
            )
            return

        # Создание эмбеддингов для текстов
        try:
            embeddings = embedding_model.encode(news_df["cleaned_text"].tolist())
            logger.info(f"Task {task_id}: Эмбеддинги созданы")
        except Exception as e:
            error_message = f"Ошибка при создании эмбеддингов: {str(e)}"
            logger.error(f"Task {task_id}: {error_message}")
            save_clustering_result(
                {
                    "id": task_id,
                    "user_id": user_id,
                    "status": "failed",
                    "date_range": {
                        "start_date": request.date_range.start_date,
                        "end_date": request.date_range.end_date,
                    },
                    "error": error_message,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ml_config": ml_config,
                    "credits_used": 0,
                }
            )
            return

        # Снижение размерности с использованием пользовательской конфигурации
        try:
            reduction_model = ReductionModel(**reduction_model_config)
            reduced_embeddings = reduction_model.fit_transform(embeddings)
            logger.info(f"Task {task_id}: Размерность снижена")
        except Exception as e:
            error_message = f"Ошибка при снижении размерности: {str(e)}"
            logger.error(f"Task {task_id}: {error_message}")
            save_clustering_result(
                {
                    "id": task_id,
                    "user_id": user_id,
                    "status": "failed",
                    "date_range": {
                        "start_date": request.date_range.start_date,
                        "end_date": request.date_range.end_date,
                    },
                    "error": error_message,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ml_config": ml_config,
                    "credits_used": 0,
                }
            )
            return

        # Кластеризация с использованием пользовательской конфигурации
        try:
            clustering_model = ClusteringModel(**clustering_model_config)
            clusters = clustering_model.fit_predict(reduced_embeddings)
            logger.info(f"Task {task_id}: Кластеризация выполнена")
        except Exception as e:
            error_message = f"Ошибка при кластеризации: {str(e)}"
            logger.error(f"Task {task_id}: {error_message}")
            save_clustering_result(
                {
                    "id": task_id,
                    "user_id": user_id,
                    "status": "failed",
                    "date_range": {
                        "start_date": request.date_range.start_date,
                        "end_date": request.date_range.end_date,
                    },
                    "error": error_message,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ml_config": ml_config,
                    "credits_used": 0,
                }
            )
            return

        # Создание представления топиков
        class TopicModel:
            def __init__(self):
                self.topics_ = clusters
                self.documents = news_df["cleaned_text"].tolist()

            def get_topic_info(self):
                topics = pd.DataFrame({"Topic": list(set(clusters))})
                return topics

            def get_representative_docs(self, topic_id):
                indices = [i for i, t in enumerate(clusters) if t == topic_id]
                return [self.documents[i] for i in indices]

            def get_topic(self, topic_id):
                indices = [i for i, t in enumerate(clusters) if t == topic_id]
                docs = [self.documents[i] for i in indices]
                words = " ".join(docs).split()
                word_count = {}
                for word in words:
                    if word not in word_count:
                        word_count[word] = 0
                    word_count[word] += 1
                sorted_words = sorted(
                    word_count.items(), key=lambda x: x[1], reverse=True
                )
                return sorted_words[:10]

            def set_topic_labels(self, topic_names):
                self.topic_labels = topic_names

        topic_model = TopicModel()

        try:
            topic_names = create_topic_names(topic_model, embedding_model)
            logger.info(f"Task {task_id}: Названия топиков созданы")
        except Exception as e:
            logger.warning(
                f"Task {task_id}: Ошибка при создании названий топиков: {e}. Используем дефолтные названия."
            )
            topic_names = {i: f"Топик {i}" for i in set(clusters) if i != -1}

        # Добавляем кластеры в оригинальный датафрейм
        news_df["cluster"] = clusters

        # Извлечение компаний из текстов
        known_companies = [
            "Газпром",
            "Роснефть",
            "Сбербанк",
            "ВТБ",
            "Яндекс",
            "Mail.ru Group",
            "Аэрофлот",
            "РЖД",
            "МТС",
            "Мегафон",
            "Билайн",
            "Тинькофф",
            "Магнит",
            "X5 Retail Group",
            "Лукойл",
            "Норникель",
            "Русал",
            "Северсталь",
            "Росатом",
            "Ростех",
            "Альфа-банк",
            "Райффайзенбанк",
            "Ozon",
            "Wildberries",
        ]

        import re

        def extract_companies(text):
            companies = []
            for company in known_companies:
                if re.search(r"\b" + re.escape(company) + r"\b", text, re.IGNORECASE):
                    companies.append(company)
            return companies

        # Извлекаем компании из текстов
        news_df["companies"] = news_df["text"].apply(extract_companies)

        # Создаем словарь компаний по кластерам
        clusters_data = {}

        for cluster_id in set(clusters):
            if cluster_id == -1:  # Пропускаем выбросы
                continue

            cluster_name = topic_names.get(cluster_id, f"Кластер {cluster_id}")

            # Получаем все новости в данном кластере
            cluster_news = news_df[news_df["cluster"] == cluster_id]

            # Собираем все компании в этом кластере
            cluster_companies = set()
            for companies_list in cluster_news["companies"]:
                cluster_companies.update(companies_list)

            # Собираем информацию о компаниях
            companies_info = []
            for company in cluster_companies:
                # Подсчитываем количество упоминаний
                company_mentions = sum(
                    1 for companies in cluster_news["companies"] if company in companies
                )

                # Получаем последнюю новость с упоминанием компании
                company_news = cluster_news[
                    cluster_news["companies"].apply(lambda x: company in x)
                ]
                if not company_news.empty:
                    latest_news = company_news.sort_values(
                        "date", ascending=False
                    ).iloc[0]
                    latest_news_text = latest_news["text"]
                    latest_news_date = latest_news["date"].strftime("%Y-%m-%d")
                else:
                    latest_news_text = ""
                    latest_news_date = ""

                companies_info.append(
                    {
                        "name": company,
                        "mentions": company_mentions,
                        "latest_news": latest_news_text,
                        "latest_news_date": latest_news_date,
                    }
                )

            # Сортируем компании по количеству упоминаний
            companies_info.sort(key=lambda x: x["mentions"], reverse=True)

            # Добавляем информацию о кластере
            clusters_data[cluster_id] = {
                "id": cluster_id,
                "name": cluster_name,
                "companies": companies_info,
                "news_count": len(cluster_news),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        # Создаем визуализацию (для сохранения в файл)
        try:
            # Используем UMAP для снижения размерности до 2D для визуализации
            umap_2d = ReductionModel(
                method="umap",
                n_neighbors=15,
                n_components=2,
                min_dist=0.1,
                metric="euclidean",
            )
            embeddings_2d = umap_2d.fit_transform(embeddings)

            # Сохраняем координаты для последующей визуализации
            visualization_data = {
                "x": embeddings_2d[:, 0].tolist(),
                "y": embeddings_2d[:, 1].tolist(),
                "cluster": clusters.tolist(),
                "text": news_df["text"].tolist(),
                "cluster_names": {str(k): v for k, v in topic_names.items()},
            }

            # Создаем директорию для сохранения визуализации
            os.makedirs("data/processed", exist_ok=True)

            # Сохраняем визуализацию для последующего использования
            viz_path = f"data/processed/viz_{task_id}.json"
            with open(viz_path, "w", encoding="utf-8") as f:
                safe_json_dump(visualization_data, f, ensure_ascii=False)

            logger.info(f"Task {task_id}: Визуализация сохранена")

        except Exception as e:
            logger.error(f"Task {task_id}: Ошибка при создании визуализации: {e}")

        # Сохраняем информацию об использованных моделях
        used_ml_config = {
            "embedding": embedding_model_config,
            "reduction": reduction_model_config,
            "clustering": clustering_model_config,
        }

        # Списываем кредиты только при успешном выполнении задачи
        user = get_user_by_id(user_id)
        if user:
            update_user_credits(
                user_id,
                user["credits"] - credits_cost,
                transaction_type="usage",
                description=f"Кластеризация успешно выполнена (ID: {task_id})",
            )

        # Сохраняем результаты кластеризации
        result = {
            "id": task_id,
            "user_id": user_id,
            "status": "completed",
            "date_range": {
                "start_date": request.date_range.start_date,
                "end_date": request.date_range.end_date,
            },
            "clusters": [clusters_data[cluster_id] for cluster_id in clusters_data],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "visualization_path": f"viz_{task_id}.json",
            "ml_config": used_ml_config,
            "credits_used": credits_cost,
        }

        save_clustering_result(result)
        logger.info(f"Task {task_id}: Кластеризация успешно завершена")

    except Exception as e:
        # В случае ошибки сохраняем информацию об ошибке, но НЕ списываем кредиты
        error_message = str(e)
        logger.error(
            f"Task {task_id}: Критическая ошибка при кластеризации: {error_message}"
        )

        error_result = {
            "id": task_id,
            "user_id": user_id,
            "status": "failed",
            "date_range": {
                "start_date": request.date_range.start_date,
                "end_date": request.date_range.end_date,
            },
            "error": error_message,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "credits_used": 0,  # При ошибке кредиты не списываются
        }
        save_clustering_result(error_result)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
