from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

from database.database import get_user_by_username

# Загрузка переменных окружения
load_dotenv()

# Настройки безопасности
SECRET_KEY = os.getenv("SECRET_KEY", "defaultsecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Проверяем, что SECRET_KEY установлен и не является дефолтным значением
if SECRET_KEY == "defaultsecretkey":
    import warnings

    warnings.warn(
        "Используется дефолтный SECRET_KEY! Это небезопасно для продакшена. "
        "Установите переменную окружения SECRET_KEY.",
        UserWarning,
    )

# Создаем контекст для хеширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Создаем схему проверки токена
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

# Список пользователей-администраторов (можно хранить в базе данных)
ADMIN_USERS = os.getenv("ADMIN_USERS", "admin").split(",")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет соответствие пароля его хешу"""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Создает хеш пароля"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Создает JWT токен доступа"""
    to_encode = data.copy()

    # Используем timezone-aware datetime вместо deprecated utcnow()
    now = datetime.now(timezone.utc)

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire.timestamp()})

    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при создании токена",
        )


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Получает текущего пользователя из JWT токена"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Недействительные учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

        # Проверяем срок действия токена
        exp = payload.get("exp")
        if exp is None:
            raise credentials_exception

        # Сравниваем с текущим временем
        current_time = datetime.now(timezone.utc).timestamp()
        if current_time > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Токен истек",
                headers={"WWW-Authenticate": "Bearer"},
            )

    except JWTError:
        raise credentials_exception
    except Exception:
        raise credentials_exception

    try:
        user = get_user_by_username(username)
        if user is None:
            raise credentials_exception
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при получении информации о пользователе",
        )


async def get_current_admin_user(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Проверяет, является ли текущий пользователь администратором.

    Зависит от get_current_user для получения информации о пользователе
    из токена аутентификации.

    Возвращает:
        dict: Информация о пользователе, если он администратор

    Исключения:
        HTTPException: Если пользователь не является администратором
    """
    if current_user["username"] not in ADMIN_USERS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуются права администратора",
        )
    return current_user
