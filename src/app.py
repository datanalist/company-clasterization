from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any
import uvicorn
import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import json
import uuid
from pathlib import Path
import asyncio

from .ml_core.models import (
    EmbeddingModel,
    ReductionModel,
    ClusteringModel,
    VectorizerModel,
    RepresentationModel,
    create_topic_names
)
from .ml_core.cluster_plot import create_interactive_topic_plot
from .data.prepare import clean_text, tokenize_ru
from .models.models import (
    ClusteringRequest,
    ClusteringResponse,
    ClusteringResult,
    ClusterInfo,
    CompanyInfo,
    DateRangeFilter,
    User,
    UserCreate
)
from .database.database import (
    get_db_connection,
    initialize_database,
    add_user,
    get_user_by_username,
    update_user_credits,
    save_clustering_result,
    get_clustering_results,
    get_clustering_result_by_id
)
from .utils.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user
)

# Создаем директории если их нет
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Инициализируем базу данных
initialize_database()

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
    db_user = get_user_by_username(user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Пользователь уже зарегистрирован")
    
    hashed_password = get_password_hash(user.password)
    user_data = user.dict()
    user_data.pop("password")
    user_data["hashed_password"] = hashed_password
    user_data["credits"] = 100  # Начальное количество кредитов
    
    return add_user(user_data)

@app.post("/api/token")
def login_for_access_token(username: str, password: str):
    user = get_user_by_username(username)
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/users/me", response_model=User)
def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user

@app.post("/api/clustering/start")
async def start_clustering(
    request: ClusteringRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    # Проверяем кредиты пользователя
    if current_user["credits"] < 10:  # Стоимость кластеризации
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Недостаточно кредитов"
        )
    
    # Создаем уникальный идентификатор для задачи
    task_id = str(uuid.uuid4())
    
    # Запускаем задачу кластеризации в фоновом режиме
    background_tasks.add_task(
        perform_clustering_task,
        task_id=task_id,
        request=request,
        user_id=current_user["id"]
    )
    
    # Списываем кредиты пользователя
    update_user_credits(current_user["id"], current_user["credits"] - 10)
    
    return {"task_id": task_id, "status": "started"}

@app.get("/api/clustering/results", response_model=List[ClusteringResult])
def get_user_clustering_results(current_user: dict = Depends(get_current_user)):
    results = get_clustering_results(user_id=current_user["id"])
    return results

@app.get("/api/clustering/results/{task_id}", response_model=ClusteringResult)
def get_clustering_result(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    result = get_clustering_result_by_id(task_id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Результат не найден"
        )
    
    if result["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Нет доступа к этому результату"
        )
    
    return result

async def perform_clustering_task(task_id: str, request: ClusteringRequest, user_id: int):
    try:
        # Загрузка новостей
        from data.parse_tg import parse_forbes_news
        from data.parse_lenta import parse_lenta_news
        
        # Определяем период времени для анализа
        start_date = datetime.strptime(request.date_range.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.date_range.end_date, "%Y-%m-%d")
        days = (end_date - start_date).days + 1
        
        # Получаем данные
        # Используем asyncio.to_thread для запуска синхронных функций асинхронно
        forbes_news_future = asyncio.create_task(
            asyncio.to_thread(lambda: asyncio.run(parse_forbes_news(days=days)))
        )
        lenta_news_future = asyncio.create_task(
            asyncio.to_thread(lambda: parse_lenta_news(days=days))
        )
        
        # Ждем выполнения обоих задач
        forbes_df = await forbes_news_future
        lenta_df = await lenta_news_future
        
        # Объединяем данные
        forbes_df = forbes_df.rename(columns={"text": "text", "date": "date"})
        forbes_df["source"] = "forbes"
        lenta_df = lenta_df.rename(columns={"text": "text", "date": "date"})
        lenta_df["source"] = "lenta"
        
        # Объединение данных
        news_df = pd.concat([forbes_df, lenta_df], ignore_index=True)
        
        # Фильтруем по дате
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = news_df[
            (news_df['date'] >= start_date) &
            (news_df['date'] <= end_date)
        ]
        
        # Очистка текста
        news_df['cleaned_text'] = news_df['text'].apply(clean_text)
        
        # Загрузка модели эмбеддингов
        embedding_model = EmbeddingModel(
            method="sentence_transformer",
            model_name_or_path="DeepPavlov/rubert-base-cased"
        )
        
        # Создание эмбеддингов для текстов
        embeddings = embedding_model.encode(news_df['cleaned_text'].tolist())
        
        # Снижение размерности
        reduction_model = ReductionModel(
            method="umap",
            n_neighbors=15,
            n_components=100,
            min_dist=0.0,
            metric='euclidean'
        )
        reduced_embeddings = reduction_model.fit_transform(embeddings)
        
        # Кластеризация
        clustering_model = ClusteringModel(
            model_name="hdbscan",
            min_cluster_size=15,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        clusters = clustering_model.fit_predict(reduced_embeddings)
        
        # Векторизатор для выделения ключевых слов
        vectorizer = VectorizerModel(
            model_name="count",
            stop_words="russian"
        )
        
        # Создание представления топиков
        # Имитируем BERTopic для создания имен топиков
        class TopicModel:
            def __init__(self):
                self.topics_ = clusters
                self.documents = news_df['cleaned_text'].tolist()
                
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
                sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
                return sorted_words[:10]
                
            def set_topic_labels(self, topic_names):
                self.topic_labels = topic_names
        
        topic_model = TopicModel()
        topic_names = create_topic_names(topic_model, embedding_model)
        
        # Добавляем кластеры в оригинальный датафрейм
        news_df['cluster'] = clusters
        
        # Извлечение компаний из текстов
        # Это упрощенная версия, в реальном проекте нужен более продвинутый NER
        import re
        
        # Список известных компаний (можно расширить)
        known_companies = [
            "Газпром", "Роснефть", "Сбербанк", "ВТБ", "Яндекс", "Mail.ru Group", 
            "Аэрофлот", "РЖД", "МТС", "Мегафон", "Билайн", "Тинькофф", "Магнит", 
            "X5 Retail Group", "Лукойл", "Норникель", "Русал", "Северсталь", 
            "Росатом", "Ростех", "Альфа-банк", "Райффайзенбанк", "Ozon", "Wildberries"
        ]
        
        def extract_companies(text):
            companies = []
            for company in known_companies:
                if re.search(r'\b' + re.escape(company) + r'\b', text, re.IGNORECASE):
                    companies.append(company)
            return companies
        
        # Извлекаем компании из текстов
        news_df['companies'] = news_df['text'].apply(extract_companies)
        
        # Создаем словарь компаний по кластерам
        clusters_data = {}
        
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Пропускаем выбросы
                continue
                
            cluster_name = topic_names.get(cluster_id, f"Кластер {cluster_id}")
            
            # Получаем все новости в данном кластере
            cluster_news = news_df[news_df['cluster'] == cluster_id]
            
            # Собираем все компании в этом кластере
            cluster_companies = set()
            for companies_list in cluster_news['companies']:
                cluster_companies.update(companies_list)
            
            # Собираем информацию о компаниях
            companies_info = []
            for company in cluster_companies:
                # Подсчитываем количество упоминаний
                company_mentions = sum(1 for companies in cluster_news['companies'] if company in companies)
                
                # Получаем последнюю новость с упоминанием компании
                company_news = cluster_news[cluster_news['companies'].apply(lambda x: company in x)]
                if not company_news.empty:
                    latest_news = company_news.sort_values('date', ascending=False).iloc[0]
                    latest_news_text = latest_news['text']
                    latest_news_date = latest_news['date'].strftime("%Y-%m-%d")
                else:
                    latest_news_text = ""
                    latest_news_date = ""
                
                companies_info.append({
                    "name": company,
                    "mentions": company_mentions,
                    "latest_news": latest_news_text,
                    "latest_news_date": latest_news_date
                })
            
            # Сортируем компании по количеству упоминаний
            companies_info.sort(key=lambda x: x["mentions"], reverse=True)
            
            # Добавляем информацию о кластере
            clusters_data[cluster_id] = {
                "id": cluster_id,
                "name": cluster_name,
                "companies": companies_info,
                "news_count": len(cluster_news),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Создаем визуализацию (для сохранения в файл)
        try:
            # Используем UMAP для снижения размерности до 2D для визуализации
            umap_2d = ReductionModel(
                method="umap",
                n_neighbors=15,
                n_components=2,
                min_dist=0.1,
                metric='euclidean'
            )
            embeddings_2d = umap_2d.fit_transform(embeddings)
            
            # Сохраняем координаты для последующей визуализации
            visualization_data = {
                "x": embeddings_2d[:, 0].tolist(),
                "y": embeddings_2d[:, 1].tolist(),
                "cluster": clusters.tolist(),
                "text": news_df['text'].tolist(),
                "cluster_names": {str(k): v for k, v in topic_names.items()}
            }
            
            # Создаем директорию для сохранения визуализации
            os.makedirs("data/processed", exist_ok=True)
            
            # Сохраняем визуализацию для последующего использования
            viz_path = f"data/processed/viz_{task_id}.json"
            with open(viz_path, 'w') as f:
                json.dump(visualization_data, f)
                
        except Exception as e:
            print(f"Ошибка при создании визуализации: {e}")
        
        # Сохраняем результаты кластеризации
        result = {
            "id": task_id,
            "user_id": user_id,
            "status": "completed",
            "date_range": {
                "start_date": request.date_range.start_date,
                "end_date": request.date_range.end_date
            },
            "clusters": [clusters_data[cluster_id] for cluster_id in clusters_data],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "visualization_path": f"viz_{task_id}.json"
        }
        
        save_clustering_result(result)
        
    except Exception as e:
        # В случае ошибки сохраняем информацию об ошибке
        error_result = {
            "id": task_id,
            "user_id": user_id,
            "status": "failed",
            "date_range": {
                "start_date": request.date_range.start_date,
                "end_date": request.date_range.end_date
            },
            "error": str(e),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_clustering_result(error_result)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
