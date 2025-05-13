from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime


class DateRangeFilter(BaseModel):
    start_date: str
    end_date: str


class ClusteringRequest(BaseModel):
    date_range: DateRangeFilter


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


class ClusteringResponse(BaseModel):
    task_id: str
    status: str


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