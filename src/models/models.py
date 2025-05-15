from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime


class DateRangeFilter(BaseModel):
    start_date: str
    end_date: str


class ClusteringRequest(BaseModel):
    date_range: DateRangeFilter
    ml_config: Optional[Dict[str, Any]] = None


class CompanyInfo(BaseModel):
    name: str
    mentions: int
    latest_news: Optional[str] = None
    latest_news_date: Optional[str] = None


class ClusterInfo(BaseModel):
    id: int
    name: str
    companies: List[CompanyInfo]
    news_count: int
    created_at: str


class ClusteringResult(BaseModel):
    id: str
    user_id: int
    status: str
    date_range: DateRangeFilter
    clusters: Optional[List[ClusterInfo]] = None
    error: Optional[str] = None
    created_at: str
    visualization_path: Optional[str] = None
    ml_config: Optional[Dict[str, Any]] = None
    credits_used: int = 10
    progress: float = 0


class ClusteringResponse(BaseModel):
    task_id: str
    status: str
    progress: float = 0


class UserCreate(BaseModel):
    username: str
    password: str
    email: str


class User(BaseModel):
    id: int
    username: str
    email: str
    credits: int
    created_at: str


# Новые модели для работы с ML-моделями
class MLModelConfig(BaseModel):
    id: Optional[int] = None
    name: str
    type: str  # embedding, reduction, clustering
    config: Dict[str, Any]
    is_default: bool = False
    created_at: Optional[str] = None
    

class MLModelUpload(BaseModel):
    name: str
    type: str
    config: Dict[str, Any]
    is_default: bool = False


class MLModelsList(BaseModel):
    embedding_models: List[MLModelConfig]
    reduction_models: List[MLModelConfig]
    clustering_models: List[MLModelConfig]


# Модели для пополнения кредитов
class CreditPackage(BaseModel):
    id: int
    name: str
    credits: int
    price: float
    description: Optional[str] = None


class CreditPurchase(BaseModel):
    package_id: int
    payment_method: str


# Модели для аналитики
class UserStats(BaseModel):
    total_clustering_tasks: int
    successful_tasks: int
    failed_tasks: int
    credits_spent: int
    avg_clusters_per_task: float


class SystemStats(BaseModel):
    total_users: int
    active_users: int
    total_tasks: int
    credits_spent: int
    popular_models: Dict[str, int] 


# Модели для математических задач
class MathProblem(BaseModel):
    id: str
    problem: str
    expires_at: str


class MathSolution(BaseModel):
    problem_id: str
    answer: float 