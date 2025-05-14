import subprocess
import sys
import os
import signal
import time
import argparse
from threading import Thread
import importlib.util

def run_backend(host="0.0.0.0", port=8000, reload=True):
    """Запускает бэкенд FastAPI"""
    cmd = [
        sys.executable, 
        "-m", "uvicorn", 
        "src.app:app", 
        "--host", host, 
        "--port", str(port),
        "--log-level", "warning"  # Снижаем уровень логирования
    ]
    
    if reload:
        cmd.append("--reload")
        
    backend_process = subprocess.Popen(cmd)
    return backend_process

def run_frontend(host="localhost", port=8501):
    """Запускает фронтенд Streamlit"""
    cmd = [
        sys.executable, 
        "-m", "streamlit", 
        "run", 
        "src/frontend/app.py",
        "--server.address", host,
        "--server.port", str(port)
    ]
    
    frontend_process = subprocess.Popen(cmd)
    return frontend_process

def print_info(backend_port, frontend_port):
    """Выводит информацию о запущенных сервисах"""
    print("\n" + "=" * 50)
    print("Система кластеризации компаний запущена!")
    print("=" * 50)
    print(f"Бэкенд FastAPI:    http://localhost:{backend_port}")
    print(f"API документация:  http://localhost:{backend_port}/docs")
    print(f"Фронтенд Streamlit: http://localhost:{frontend_port}")
    print("\nДля остановки нажмите Ctrl+C\n")
    print("=" * 50 + "\n")

def check_requirements():
    """Проверяет, установлены ли все необходимые зависимости"""
    required_packages = [
        "fastapi",
        "streamlit",
        "pandas",
        "nltk",
        "sentence_transformers",
        "umap",
        "hdbscan"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Следующие пакеты не найдены: {', '.join(missing_packages)}")
        print("\nПожалуйста, установите все зависимости с помощью команды:")
        print("pip install -r requirements.txt")
        return False
    return True

def create_directories():
    """Создает необходимые директории, если они не существуют"""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

def run_migration():
    """Запускает миграцию базы данных"""
    print("Запуск миграции базы данных...")
    
    try:
        # Импортируем и запускаем функцию миграции
        from src.database.database import initialize_database
        initialize_database()
        print("Миграция базы данных успешно выполнена")
        return True
    except Exception as e:
        print(f"Ошибка при миграции базы данных: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск системы кластеризации компаний")
    parser.add_argument("--backend-host", default="0.0.0.0", help="Хост для бэкенда FastAPI")
    parser.add_argument("--backend-port", type=int, default=8000, help="Порт для бэкенда FastAPI")
    parser.add_argument("--frontend-host", default="localhost", help="Хост для фронтенда Streamlit")
    parser.add_argument("--frontend-port", type=int, default=8501, help="Порт для фронтенда Streamlit")
    parser.add_argument("--no-reload", action="store_true", help="Отключить автоматическую перезагрузку бэкенда")
    parser.add_argument("--skip-check", action="store_true", help="Пропустить проверку зависимостей")
    parser.add_argument("--migrate", action="store_true", help="Выполнить миграцию базы данных перед запуском")
    
    args = parser.parse_args()
    
    # Проверяем зависимости, если не указан флаг --skip-check
    if not args.skip_check and not check_requirements():
        sys.exit(1)
    
    # Создаем необходимые директории
    create_directories()
    
    # Запускаем миграцию базы данных, если указан флаг --migrate
    if args.migrate and not run_migration():
        print("Миграция базы данных завершилась с ошибкой. Продолжить запуск? (y/n)")
        response = input().lower()
        if response != 'y':
            print("Выход...")
            sys.exit(1)
    
    processes = []
    
    try:
        # Запускаем бэкенд
        backend_process = run_backend(
            host=args.backend_host, 
            port=args.backend_port, 
            reload=not args.no_reload
        )
        processes.append(backend_process)
        
        # Даем время на запуск бэкенда
        time.sleep(2)
        
        # Запускаем фронтенд
        frontend_process = run_frontend(
            host=args.frontend_host, 
            port=args.frontend_port
        )
        processes.append(frontend_process)
        
        # Выводим информацию
        print_info(args.backend_port, args.frontend_port)
        
        # Ждем завершения процессов
        while all(p.poll() is None for p in processes):
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nЗавершение работы...")
    finally:
        # Завершаем все процессы
        for p in processes:
            if p.poll() is None:  # процесс еще работает
                if sys.platform == 'win32':
                    p.send_signal(signal.CTRL_C_EVENT)
                else:
                    p.send_signal(signal.SIGINT)
                
                # Даем время на корректное завершение
                time.sleep(1)
                
                # Если процесс все еще работает, принудительно завершаем
                if p.poll() is None:
                    p.terminate()
        
        print("Система остановлена.") 