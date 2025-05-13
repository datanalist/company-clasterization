import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

# URL бэкенда API
API_URL = "http://localhost:8000"

# Настройка страницы
st.set_page_config(
    page_title="Кластеризация компаний по новостям",
    page_icon="📊",
    layout="wide",
)

# Стили CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .cluster-card {
        background-color: #f1f1f1;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .cluster-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .company-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
    }
    .status-completed {
        color: green;
        font-weight: bold;
    }
    .status-failed {
        color: red;
        font-weight: bold;
    }
    .status-running {
        color: orange;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Функции для работы с API
def register_user(username, password, email):
    try:
        response = requests.post(
            f"{API_URL}/api/register",
            json={"username": username, "password": password, "email": email}
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def login_user(username, password):
    try:
        response = requests.post(
            f"{API_URL}/api/token",
            params={"username": username, "password": password}
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def get_user_info(token):
    try:
        response = requests.get(
            f"{API_URL}/api/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def start_clustering(token, date_range):
    try:
        response = requests.post(
            f"{API_URL}/api/clustering/start",
            headers={"Authorization": f"Bearer {token}"},
            json={"date_range": date_range}
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def get_clustering_results(token):
    try:
        response = requests.get(
            f"{API_URL}/api/clustering/results",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def get_clustering_result(token, task_id):
    try:
        response = requests.get(
            f"{API_URL}/api/clustering/results/{task_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

# Проверка авторизации
def check_authentication():
    if "token" not in st.session_state:
        show_login_page()
        return False
    
    # Проверка валидности токена
    user_info, status_code = get_user_info(st.session_state["token"])
    if status_code != 200:
        # Токен невалидный, показываем страницу логина
        st.session_state.pop("token", None)
        st.session_state.pop("username", None)
        show_login_page()
        return False
    
    st.session_state["user_info"] = user_info
    return True

# Страница авторизации
def show_login_page():
    st.markdown("<h1 class='main-header'>Кластеризация компаний по новостям</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Вход", "Регистрация"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Имя пользователя")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти")
            
            if submitted:
                if not username or not password:
                    st.error("Пожалуйста, заполните все поля")
                else:
                    result, status_code = login_user(username, password)
                    if status_code == 200:
                        st.session_state["token"] = result["access_token"]
                        st.session_state["username"] = username
                        st.success("Вход выполнен успешно!")
                        st.rerun()
                    else:
                        st.error(f"Ошибка: {result.get('detail', 'Неизвестная ошибка')}")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Имя пользователя")
            new_password = st.text_input("Пароль", type="password")
            confirm_password = st.text_input("Подтвердите пароль", type="password")
            email = st.text_input("Email")
            reg_submitted = st.form_submit_button("Зарегистрироваться")
            
            if reg_submitted:
                if not new_username or not new_password or not confirm_password or not email:
                    st.error("Пожалуйста, заполните все поля")
                elif new_password != confirm_password:
                    st.error("Пароли не совпадают")
                else:
                    result, status_code = register_user(new_username, new_password, email)
                    if status_code == 200:
                        st.success("Регистрация успешна! Теперь вы можете войти.")
                    else:
                        st.error(f"Ошибка: {result.get('detail', 'Неизвестная ошибка')}")

# Функция для отображения визуализации кластеров
def show_cluster_visualization(task_id):
    viz_path = f"data/processed/viz_{task_id}.json"
    
    if not os.path.exists(viz_path):
        st.warning("Визуализация недоступна для этого результата.")
        return
    
    try:
        with open(viz_path, 'r') as f:
            viz_data = json.load(f)
        
        # Создаем DataFrame для визуализации
        df = pd.DataFrame({
            'x': viz_data['x'],
            'y': viz_data['y'],
            'cluster': viz_data['cluster'],
            'text': viz_data['text']
        })
        
        # Добавляем имена кластеров
        cluster_names = viz_data['cluster_names']
        df['cluster_name'] = df['cluster'].astype(str).map(
            lambda x: cluster_names.get(x, f"Кластер {x}")
        )
        
        # Создаем график
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster_name',
            hover_data=['text'],
            title="Визуализация кластеров компаний",
            template="plotly_white",
            height=800
        )
        
        fig.update_traces(
            marker=dict(size=8, opacity=0.7),
            selector=dict(mode='markers')
        )
        
        fig.update_layout(
            legend_title_text="Кластеры",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Ошибка при загрузке визуализации: {e}")

# Функция для отображения детальной информации о кластере
def show_cluster_details(cluster):
    with st.expander(f"{cluster['name']} (Новостей: {cluster['news_count']})"):
        # Таблица компаний
        companies_data = []
        for company in cluster["companies"]:
            companies_data.append({
                "Компания": company["name"],
                "Упоминаний": company["mentions"],
                "Последняя новость": company.get("latest_news", "")[:100] + "..." if company.get("latest_news") else "",
                "Дата": company.get("latest_news_date", "")
            })
        
        if companies_data:
            df = pd.DataFrame(companies_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("В этом кластере нет компаний.")

# Главная страница приложения
def show_main_page():
    if not check_authentication():
        return
    
    user_info = st.session_state.get("user_info", {})
    
    # Боковая панель с информацией пользователя
    with st.sidebar:
        st.markdown(f"### Привет, {user_info.get('username', '')}!")
        st.markdown(f"**Кредиты:** {user_info.get('credits', 0)}")
        
        if st.button("Выйти"):
            st.session_state.pop("token", None)
            st.session_state.pop("username", None)
            st.session_state.pop("user_info", None)
            st.rerun()
    
    st.markdown("<h1 class='main-header'>Кластеризация компаний по новостям</h1>", unsafe_allow_html=True)
    
    # Секция для создания новой кластеризации
    st.markdown("### Создать новую кластеризацию")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Начальная дата",
            value=datetime.now() - timedelta(days=7)
        )
    with col2:
        end_date = st.date_input(
            "Конечная дата",
            value=datetime.now()
        )
    
    if st.button("Начать кластеризацию", type="primary"):
        if user_info.get('credits', 0) < 10:
            st.error("Недостаточно кредитов для запуска кластеризации.")
        else:
            with st.spinner("Запуск кластеризации..."):
                date_range = {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d")
                }
                result, status_code = start_clustering(st.session_state["token"], date_range)
                
                if status_code == 200:
                    st.success(f"Кластеризация запущена! ID задачи: {result['task_id']}")
                    # Обновляем информацию о пользователе (кредиты уменьшились)
                    user_info, _ = get_user_info(st.session_state["token"])
                    st.session_state["user_info"] = user_info
                else:
                    st.error(f"Ошибка: {result.get('detail', 'Неизвестная ошибка')}")
    
    # Секция с результатами кластеризации
    st.markdown("### Результаты кластеризации")
    
    if st.button("Обновить результаты"):
        st.rerun()
    
    # Получаем результаты кластеризации
    results, status_code = get_clustering_results(st.session_state["token"])
    
    if status_code != 200:
        st.error("Не удалось получить результаты кластеризации.")
        return
    
    if not results:
        st.info("У вас пока нет результатов кластеризации.")
        return
    
    # Отображение результатов
    for result in results:
        with st.expander(f"Результат от {result['created_at']} ({result['date_range']['start_date']} - {result['date_range']['end_date']})"):
            status = result["status"]
            if status == "completed":
                st.markdown(f"<p class='status-completed'>Статус: Завершено</p>", unsafe_allow_html=True)
                
                # Показываем визуализацию
                show_cluster_visualization(result["id"])
                
                # Показываем детали кластеров
                if result.get("clusters"):
                    for cluster in result["clusters"]:
                        show_cluster_details(cluster)
                else:
                    st.warning("Не найдены кластеры или компании в результате.")
                    
            elif status == "failed":
                st.markdown(f"<p class='status-failed'>Статус: Ошибка</p>", unsafe_allow_html=True)
                st.error(f"Причина: {result.get('error', 'Неизвестная ошибка')}")
            else:
                st.markdown(f"<p class='status-running'>Статус: Выполняется</p>", unsafe_allow_html=True)
                st.info("Кластеризация в процессе. Обновите страницу через некоторое время.")

# Запуск приложения
if __name__ == "__main__":
    if "token" in st.session_state:
        show_main_page()
    else:
        show_login_page() 