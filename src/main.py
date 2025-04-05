from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional, List, Dict
import uvicorn

from auth_service import (
    Token, User, create_access_token, verify_token,
    verify_password, ACCESS_TOKEN_EXPIRE_MINUTES
)
from model_service import model_service, ModelResult
from datetime import timedelta

app = FastAPI(title="Company Clasterization API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Временное хранилище пользователей (в реальном приложении использовать базу данных)
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "test@example.com",
        "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LcdYGI4oDtU2lGW/2",  # password: testpass
        "disabled": False,
    }
}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    username = verify_token(token)
    if username is None or username not in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return fake_users_db[username]

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/")
async def root():
    return {"message": "Company Clasterization API"}

@app.get("/api/companies")
async def get_companies(current_user: str = Depends(get_current_user)):
    """Получение списка компаний"""
    return {"companies": []}

@app.get("/api/models")
async def get_models(current_user: User = Depends(get_current_user)) -> List[Dict]:
    """Получение списка доступных моделей"""
    return await model_service.get_available_models()

@app.post("/api/companies/analyze")
async def analyze_company(
    company_id: int,
    model_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Запуск анализа компании"""
    analysis_id = await model_service.analyze_company(company_id, model_id)
    return {
        "status": "analysis_started",
        "analysis_id": analysis_id,
        "company_id": company_id,
        "model_id": model_id
    }

@app.get("/api/analysis/{analysis_id}")
async def get_analysis_results(
    analysis_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Получение результатов анализа"""
    result = await model_service.get_result(analysis_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    return result.dict()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
