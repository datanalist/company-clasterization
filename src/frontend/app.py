import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL бэкенда API
API_URL = "http://localhost:8000"

# Таймаут для запросов (в секундах)
REQUEST_TIMEOUT = 30

# Настройка страницы
st.set_page_config(
    page_title="Кластеризация компаний по новостям",
    page_icon="📊",
    layout="wide",
)

# Стили CSS
st.markdown(
    """
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
    .credit-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .credit-package {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #ced4da;
    }
    .credit-balance {
        font-size: 1.5rem;
        font-weight: bold;
        color: #198754;
    }
    .credit-history-item {
        padding: 0.5rem;
        border-bottom: 1px solid #dee2e6;
    }
    .purchase {
        color: #198754;
    }
    .usage {
        color: #dc3545;
    }
    .stats-card {
        background-color: #f1f9ff;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #b8daff;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Функции для работы с API
def make_request(method: str, url: str, **kwargs):
    """Общая функция для выполнения HTTP-запросов с обработкой ошибок"""
    try:
        # Добавляем таймаут ко всем запросам
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)

        response = requests.request(method, url, **kwargs)
        return response.json(), response.status_code
    except requests.exceptions.Timeout:
        logger.error(f"Таймаут запроса к {url}")
        return {"error": "Таймаут запроса к серверу"}, 408
    except requests.exceptions.ConnectionError:
        logger.error(f"Ошибка соединения с {url}")
        return {"error": "Не удалось подключиться к серверу"}, 503
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса к {url}: {e}")
        return {"error": f"Ошибка запроса: {str(e)}"}, 500
    except json.JSONDecodeError:
        logger.error(f"Неверный JSON ответ от {url}")
        return {"error": "Неверный формат ответа сервера"}, 500
    except Exception as e:
        logger.error(f"Неожиданная ошибка при запросе к {url}: {e}")
        return {"error": f"Неожиданная ошибка: {str(e)}"}, 500


def register_user(username: str, password: str, email: str):
    """Регистрация нового пользователя"""
    return make_request(
        "POST",
        f"{API_URL}/api/register",
        json={"username": username, "password": password, "email": email},
    )


def login_user(username: str, password: str):
    """Аутентификация пользователя"""
    return make_request(
        "POST",
        f"{API_URL}/api/token",
        params={"username": username, "password": password},
    )


def get_user_info(token: str):
    """Получение информации о текущем пользователе"""
    return make_request(
        "GET", f"{API_URL}/api/users/me", headers={"Authorization": f"Bearer {token}"}
    )


def start_clustering(token: str, date_range: dict, ml_config=None):
    """Запуск задачи кластеризации"""
    data = {"date_range": date_range}
    if ml_config:
        data["ml_config"] = ml_config

    return make_request(
        "POST",
        f"{API_URL}/api/clustering/start",
        headers={"Authorization": f"Bearer {token}"},
        json=data,
    )


def get_clustering_results(token: str):
    """Получение результатов кластеризации"""
    return make_request(
        "GET",
        f"{API_URL}/api/clustering/results",
        headers={"Authorization": f"Bearer {token}"},
    )


def get_clustering_result(token: str, task_id: str):
    """Получение конкретного результата кластеризации"""
    return make_request(
        "GET",
        f"{API_URL}/api/clustering/results/{task_id}",
        headers={"Authorization": f"Bearer {token}"},
    )


def get_ml_models(token: str):
    """Получение списка доступных ML-моделей"""
    return make_request(
        "GET", f"{API_URL}/api/ml/models", headers={"Authorization": f"Bearer {token}"}
    )


def get_credit_packages(token: str):
    """Получение списка пакетов кредитов"""
    return make_request(
        "GET",
        f"{API_URL}/api/credits/packages",
        headers={"Authorization": f"Bearer {token}"},
    )


def purchase_credits(token: str, package_id: int, payment_method: str):
    """Покупка кредитов"""
    return make_request(
        "POST",
        f"{API_URL}/api/credits/purchase",
        headers={"Authorization": f"Bearer {token}"},
        json={"package_id": package_id, "payment_method": payment_method},
    )


def get_credit_history(token: str):
    """Получение истории кредитных операций"""
    return make_request(
        "GET",
        f"{API_URL}/api/credits/history",
        headers={"Authorization": f"Bearer {token}"},
    )


def get_user_stats(token: str):
    """Получение статистики пользователя"""
    return make_request(
        "GET", f"{API_URL}/api/stats/user", headers={"Authorization": f"Bearer {token}"}
    )


def get_math_problem(token: str):
    """Получение математической задачи"""
    return make_request(
        "GET",
        f"{API_URL}/api/credits/math-problem",
        headers={"Authorization": f"Bearer {token}"},
    )


def check_math_solution(token: str, problem_id: str, answer: float):
    """Проверка решения математической задачи"""
    return make_request(
        "POST",
        f"{API_URL}/api/credits/math-solution",
        headers={"Authorization": f"Bearer {token}"},
        json={"problem_id": problem_id, "answer": answer},
    )


# Проверка авторизации
def check_authentication():
    """Проверяет авторизацию пользователя"""
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
    """Отображает страницу входа и регистрации"""
    st.markdown(
        "<h1 class='main-header'>Кластеризация компаний по новостям</h1>",
        unsafe_allow_html=True,
    )

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
                    with st.spinner("Выполняется вход..."):
                        result, status_code = login_user(username, password)
                        if status_code == 200:
                            st.session_state["token"] = result["access_token"]
                            st.session_state["username"] = username
                            st.success("Вход выполнен успешно!")
                            st.rerun()
                        else:
                            st.error(
                                f"Ошибка: {result.get('detail', result.get('error', 'Неизвестная ошибка'))}"
                            )

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Имя пользователя")
            new_password = st.text_input("Пароль", type="password")
            confirm_password = st.text_input("Подтвердите пароль", type="password")
            email = st.text_input("Email")
            reg_submitted = st.form_submit_button("Зарегистрироваться")

            if reg_submitted:
                if (
                    not new_username
                    or not new_password
                    or not confirm_password
                    or not email
                ):
                    st.error("Пожалуйста, заполните все поля")
                elif new_password != confirm_password:
                    st.error("Пароли не совпадают")
                else:
                    with st.spinner("Выполняется регистрация..."):
                        result, status_code = register_user(
                            new_username, new_password, email
                        )
                        if status_code == 200:
                            st.success("Регистрация успешна! Теперь вы можете войти.")
                        else:
                            st.error(
                                f"Ошибка: {result.get('detail', result.get('error', 'Неизвестная ошибка'))}"
                            )


# Функция для отображения визуализации кластеров
def show_cluster_visualization(task_id: str):
    """Отображает визуализацию кластеров для конкретной задачи"""
    viz_path = f"data/processed/viz_{task_id}.json"

    if not os.path.exists(viz_path):
        st.warning("Визуализация недоступна для этого результата.")
        return

    try:
        with open(viz_path, encoding="utf-8") as f:
            viz_data = json.load(f)

        # Создаем DataFrame для визуализации
        df = pd.DataFrame(
            {
                "x": viz_data["x"],
                "y": viz_data["y"],
                "cluster": viz_data["cluster"],
                "text": viz_data["text"],
            }
        )

        # Добавляем имена кластеров
        cluster_names = viz_data["cluster_names"]
        df["cluster_name"] = (
            df["cluster"]
            .astype(str)
            .map(lambda x: cluster_names.get(x, f"Кластер {x}"))
        )

        # Создаем график
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster_name",
            hover_data=["text"],
            title="Визуализация кластеров компаний",
            template="plotly_white",
            height=800,
        )

        fig.update_traces(
            marker=dict(size=8, opacity=0.7), selector=dict(mode="markers")
        )

        fig.update_layout(
            legend_title_text="Кластеры",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.1),
        )

        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning("Файл визуализации не найден.")
    except json.JSONDecodeError:
        st.error("Ошибка при чтении файла визуализации.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке визуализации: {e}")
        st.error(f"Ошибка при загрузке визуализации: {e}")


# Функция для отображения детальной информации о кластере
def show_cluster_details(cluster: dict):
    """Отображает детальную информацию о кластере"""
    with st.expander(f"{cluster['name']} (Новостей: {cluster['news_count']})"):
        # Таблица компаний
        companies_data = []
        for company in cluster["companies"]:
            companies_data.append(
                {
                    "Компания": company["name"],
                    "Упоминаний": company["mentions"],
                    "Последняя новость": company.get("latest_news", "")[:100] + "..."
                    if company.get("latest_news")
                    else "",
                    "Дата": company.get("latest_news_date", ""),
                }
            )

        if companies_data:
            df = pd.DataFrame(companies_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("В этом кластере нет компаний.")


# Функция для отображения страницы управления кредитами
def show_credits_page(user_info: dict):
    """Отображает страницу управления кредитами"""
    st.markdown("<h2>Управление кредитами</h2>", unsafe_allow_html=True)

    # Отображаем текущий баланс
    st.markdown(
        f"""
    <div class="credit-card">
        <h3>Текущий баланс</h3>
        <p class="credit-balance">{user_info.get("credits", 0)} кредитов</p>
        <p>Кредиты используются для запуска задач кластеризации.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Создаем вкладки для разных способов пополнения
    tab1, tab2 = st.tabs(["Покупка кредитов", "Слава математике!"])

    with tab1:
        # Получаем доступные пакеты кредитов
        with st.spinner("Загрузка пакетов кредитов..."):
            packages, status_code = get_credit_packages(st.session_state["token"])

        if status_code == 200 and packages:
            st.markdown("<h3>Доступные пакеты кредитов</h3>", unsafe_allow_html=True)

            # Отображаем пакеты кредитов в виде карточек
            cols = st.columns(min(len(packages), 4))
            selected_package = None

            for i, package in enumerate(packages):
                with cols[i % 4]:
                    with st.container():
                        st.markdown(
                            f"""
                        <div class="credit-package">
                            <h4>{package["name"]}</h4>
                            <p><strong>{package["credits"]}</strong> кредитов</p>
                            <p>Цена: {package["price"]} руб.</p>
                            <p>{package.get("description", "")}</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        if st.button(
                            f"Купить {package['name']}", key=f"buy_{package['id']}"
                        ):
                            selected_package = package

            # Если выбран пакет для покупки
            if selected_package:
                st.markdown("<h3>Оформление покупки</h3>", unsafe_allow_html=True)

                payment_methods = ["Банковская карта", "PayPal", "Электронный кошелек"]
                payment_method = st.selectbox("Способ оплаты", payment_methods)

                if st.button("Подтвердить покупку"):
                    with st.spinner("Обработка платежа..."):
                        # Имитация задержки обработки платежа
                        time.sleep(1)

                        # Запрос на покупку кредитов
                        result, status_code = purchase_credits(
                            st.session_state["token"],
                            selected_package["id"],
                            payment_method,
                        )

                        if status_code == 200:
                            st.success(
                                result.get("message", "Кредиты успешно добавлены!")
                            )
                            # Обновляем информацию о пользователе
                            user_info, _ = get_user_info(st.session_state["token"])
                            st.session_state["user_info"] = user_info
                            st.rerun()
                        else:
                            st.error(
                                f"Ошибка: {result.get('detail', result.get('error', 'Неизвестная ошибка'))}"
                            )
        else:
            st.error(
                f"Не удалось загрузить пакеты кредитов: {packages.get('error', 'Неизвестная ошибка')}"
            )

    with tab2:
        st.markdown(
            """
        <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
            <h3>Слава математике! 🧮</h3>
            <p>Решайте математические задачи и получайте <strong>100 кредитов</strong> за каждый правильный ответ!</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Контейнер для задачи
        with st.container():
            if "math_problem" not in st.session_state:
                # Получаем новую задачу если её нет в сессии
                if st.button("Получить задачу"):
                    with st.spinner("Генерируем задачу..."):
                        problem, status_code = get_math_problem(
                            st.session_state["token"]
                        )
                        if status_code == 200:
                            st.session_state.math_problem = problem
                            st.rerun()
                        else:
                            st.error(
                                f"Не удалось получить задачу: {problem.get('error', 'Неизвестная ошибка')}"
                            )
            else:
                # Отображаем текущую задачу
                problem = st.session_state.math_problem

                st.markdown(
                    f"""
                <div style='background-color: #e9f5ff; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
                    <h2 style='font-size: 24px;'>{problem["problem"]}</h2>
                    <p>Задача действительна до: {problem["expires_at"]}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Форма для ответа
                answer = st.text_input("Введите ваш ответ:", key="math_answer")

                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("Проверить ответ"):
                        if not answer:
                            st.warning("Введите ответ!")
                        else:
                            try:
                                float_answer = float(answer.replace(",", "."))
                                with st.spinner("Проверяем ответ..."):
                                    result, status_code = check_math_solution(
                                        st.session_state["token"],
                                        problem["id"],
                                        float_answer,
                                    )

                                if status_code == 200:
                                    if result.get("success"):
                                        # Правильный ответ
                                        st.success(result.get("message"))
                                        # Обновляем информацию о пользователе
                                        user_info, _ = get_user_info(
                                            st.session_state["token"]
                                        )
                                        st.session_state["user_info"] = user_info
                                        # Удаляем задачу из сессии
                                        del st.session_state.math_problem
                                        st.rerun()
                                    else:
                                        # Неправильный ответ
                                        st.error(result.get("message"))
                                else:
                                    st.error(
                                        f"Ошибка: {result.get('detail', result.get('error', 'Неизвестная ошибка'))}"
                                    )
                            except ValueError:
                                st.error("Введите числовое значение!")

                with col2:
                    if st.button("Отменить и получить новую задачу"):
                        del st.session_state.math_problem
                        st.rerun()

    # История транзакций
    st.markdown("<h3>История транзакций</h3>", unsafe_allow_html=True)

    # Получаем историю кредитных операций
    with st.spinner("Загрузка истории транзакций..."):
        history, status_code = get_credit_history(st.session_state["token"])

    if status_code == 200 and history:
        # Создаем DataFrame для отображения истории
        try:
            history_df = pd.DataFrame(history)

            # Преобразуем дату
            if "created_at" in history_df.columns:
                history_df["created_at"] = pd.to_datetime(history_df["created_at"])
                history_df["Дата"] = history_df["created_at"].dt.strftime(
                    "%Y-%m-%d %H:%M"
                )

            # Преобразуем тип транзакции
            if "transaction_type" in history_df.columns:
                type_map = {
                    "purchase": "Покупка",
                    "usage": "Использование",
                    "refund": "Возврат",
                    "math_reward": "Награда за задачу",
                }
                history_df["Тип"] = history_df["transaction_type"].map(
                    lambda x: type_map.get(x, x)
                )

            # Выбираем нужные столбцы
            display_df = history_df[["Дата", "Тип", "amount", "description"]].rename(
                columns={"amount": "Количество", "description": "Описание"}
            )

            # Отображаем таблицу
            st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            logger.error(f"Ошибка при обработке истории транзакций: {e}")
            st.error("Ошибка при отображении истории транзакций")
    else:
        if status_code != 200:
            st.error(
                f"Ошибка при загрузке истории: {history.get('error', 'Неизвестная ошибка')}"
            )
        else:
            st.info("История транзакций пуста")


# Функция для отображения страницы с аналитикой
def show_analytics_page(user_info: dict):
    """Отображает страницу аналитики"""
    st.markdown("<h2>Аналитика использования</h2>", unsafe_allow_html=True)

    # Получаем статистику пользователя
    with st.spinner("Загрузка статистики..."):
        stats, status_code = get_user_stats(st.session_state["token"])

    if status_code == 200:
        # Отображаем общую статистику пользователя
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
            <div class="stats-card">
                <h3>Всего задач</h3>
                <p style="font-size: 2rem; font-weight: bold;">{stats.get("total_clustering_tasks", 0)}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="stats-card">
                <h3>Использовано кредитов</h3>
                <p style="font-size: 2rem; font-weight: bold;">{stats.get("credits_spent", 0)}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="stats-card">
                <h3>Успешных задач</h3>
                <p style="font-size: 2rem; font-weight: bold;">{stats.get("successful_tasks", 0)}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Создаем график распределения задач по статусам
        status_data = {
            "Статус": ["Успешно", "Ошибка"],
            "Количество": [
                stats.get("successful_tasks", 0),
                stats.get("failed_tasks", 0),
            ],
        }

        status_df = pd.DataFrame(status_data)

        if status_df["Количество"].sum() > 0:
            fig = px.pie(
                status_df,
                values="Количество",
                names="Статус",
                title="Распределение задач по статусам",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Отображаем среднее количество кластеров
        st.markdown(
            f"""
        <div class="stats-card">
            <h3>Среднее количество кластеров на задачу</h3>
            <p style="font-size: 1.5rem;">{stats.get("avg_clusters_per_task", 0):.2f}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    else:
        st.error(
            f"Не удалось получить статистику: {stats.get('error', 'Неизвестная ошибка')}"
        )


# Функция для выбора моделей ML
def show_model_selection():
    """Отображает интерфейс выбора ML-моделей"""
    st.markdown("<h3>Выбор моделей машинного обучения</h3>", unsafe_allow_html=True)

    # Получаем список доступных моделей
    with st.spinner("Загрузка списка моделей..."):
        models, status_code = get_ml_models(st.session_state["token"])

    if status_code == 200:
        # Создаем вкладки для разных типов моделей
        tab1, tab2, tab3 = st.tabs(
            [
                "Модели эмбеддингов",
                "Модели снижения размерности",
                "Модели кластеризации",
            ]
        )

        # Выбор модели для эмбеддингов
        with tab1:
            embedding_models = models.get("embedding_models", [])
            if embedding_models:
                # Находим дефолтную модель
                default_model_idx = next(
                    (i for i, m in enumerate(embedding_models) if m.get("is_default")),
                    0,
                )

                selected_embedding = st.selectbox(
                    "Выберите модель для создания эмбеддингов:",
                    embedding_models,
                    format_func=lambda x: f"{x['name']} (Дефолтная)"
                    if x.get("is_default")
                    else x["name"],
                    index=default_model_idx,
                )

                # Отображаем параметры выбранной модели
                if selected_embedding:
                    st.write("Параметры модели:")
                    st.json(selected_embedding["config"])
            else:
                st.info("Нет доступных моделей эмбеддингов")

        # Выбор модели для снижения размерности
        with tab2:
            reduction_models = models.get("reduction_models", [])
            if reduction_models:
                # Находим дефолтную модель
                default_model_idx = next(
                    (i for i, m in enumerate(reduction_models) if m.get("is_default")),
                    0,
                )

                selected_reduction = st.selectbox(
                    "Выберите модель для снижения размерности:",
                    reduction_models,
                    format_func=lambda x: f"{x['name']} (Дефолтная)"
                    if x.get("is_default")
                    else x["name"],
                    index=default_model_idx,
                )

                # Отображаем параметры выбранной модели
                if selected_reduction:
                    st.write("Параметры модели:")
                    st.json(selected_reduction["config"])
            else:
                st.info("Нет доступных моделей снижения размерности")

        # Выбор модели для кластеризации
        with tab3:
            clustering_models = models.get("clustering_models", [])
            if clustering_models:
                # Находим дефолтную модель
                default_model_idx = next(
                    (i for i, m in enumerate(clustering_models) if m.get("is_default")),
                    0,
                )

                selected_clustering = st.selectbox(
                    "Выберите модель для кластеризации:",
                    clustering_models,
                    format_func=lambda x: f"{x['name']} (Дефолтная)"
                    if x.get("is_default")
                    else x["name"],
                    index=default_model_idx,
                )

                # Отображаем параметры выбранной модели
                if selected_clustering:
                    st.write("Параметры модели:")
                    st.json(selected_clustering["config"])
            else:
                st.info("Нет доступных моделей кластеризации")

        # Если выбраны все типы моделей, возвращаем конфигурацию
        if (
            "selected_embedding" in locals()
            and "selected_reduction" in locals()
            and "selected_clustering" in locals()
        ):
            return {
                "embedding": selected_embedding["config"],
                "reduction": selected_reduction["config"],
                "clustering": selected_clustering["config"],
            }
    else:
        st.error(
            f"Не удалось получить список моделей: {models.get('error', 'Неизвестная ошибка')}"
        )

    # Возвращаем None, если не удалось получить модели или выбрать их
    return None


# Главная страница приложения
def show_main_page():
    """Отображает главную страницу приложения"""
    if not check_authentication():
        return

    user_info = st.session_state.get("user_info", {})

    # Боковая панель с информацией пользователя и навигацией
    with st.sidebar:
        st.markdown(f"### Привет, {user_info.get('username', '')}!")
        st.markdown(f"**Кредиты:** {user_info.get('credits', 0)}")

        # Навигация
        st.markdown("### Навигация")

        page = st.radio(
            "Выберите раздел:", ["Главная", "Мои кредиты", "Аналитика", "Настройки"]
        )

        if st.button("Выйти"):
            for key in ["token", "username", "user_info", "math_problem"]:
                st.session_state.pop(key, None)
            st.rerun()

    # Отображаем выбранную страницу
    if page == "Главная":
        show_clustering_page(user_info)
    elif page == "Мои кредиты":
        show_credits_page(user_info)
    elif page == "Аналитика":
        show_analytics_page(user_info)
    elif page == "Настройки":
        st.markdown("<h2>Настройки</h2>", unsafe_allow_html=True)
        st.info("Раздел находится в разработке")


# Страница кластеризации (главная)
def show_clustering_page(user_info: dict):
    """Отображает страницу кластеризации"""
    st.markdown(
        "<h1 class='main-header'>Кластеризация компаний по новостям</h1>",
        unsafe_allow_html=True,
    )

    # Секция для создания новой кластеризации
    st.markdown("### Создать новую кластеризацию")

    # Выбор периода
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Начальная дата", value=datetime.now() - timedelta(days=7)
        )
    with col2:
        end_date = st.date_input("Конечная дата", value=datetime.now())

    # Расширенные настройки (выбор моделей)
    show_advanced = st.checkbox("Показать расширенные настройки")

    ml_config = None
    if show_advanced:
        ml_config = show_model_selection()

    # Кнопка запуска кластеризации
    if st.button("Начать кластеризацию", type="primary"):
        if user_info.get("credits", 0) < 10:
            st.error("Недостаточно кредитов для запуска кластеризации.")
        else:
            with st.spinner("Запуск кластеризации..."):
                date_range = {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                }

                # Передаем конфигурацию моделей, если она выбрана
                result, status_code = start_clustering(
                    st.session_state["token"], date_range, ml_config
                )

                if status_code == 200:
                    st.success(
                        f"Кластеризация запущена! ID задачи: {result['task_id']}"
                    )
                    # Обновляем информацию о пользователе (кредиты уменьшились)
                    user_info, _ = get_user_info(st.session_state["token"])
                    st.session_state["user_info"] = user_info
                    # Автоматически обновляем страницу через несколько секунд
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(
                        f"Ошибка: {result.get('detail', result.get('error', 'Неизвестная ошибка'))}"
                    )

    # Секция с результатами кластеризации
    st.markdown("### Результаты кластеризации")

    if st.button("Обновить результаты"):
        st.rerun()

    # Получаем результаты кластеризации
    with st.spinner("Загрузка результатов..."):
        results, status_code = get_clustering_results(st.session_state["token"])

    if status_code != 200:
        st.error(
            f"Не удалось получить результаты кластеризации: {results.get('error', 'Неизвестная ошибка')}"
        )
        return

    if not results:
        st.info("У вас пока нет результатов кластеризации.")
        return

    # Отображение результатов
    for result in results:
        with st.expander(
            f"Результат от {result['created_at']} ({result['date_range']['start_date']} - {result['date_range']['end_date']})"
        ):
            status = result["status"]
            if status == "completed":
                st.markdown(
                    "<p class='status-completed'>Статус: Завершено</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"Использовано кредитов: {result.get('credits_used', 10)}")

                # Показываем информацию об использованных моделях
                if result.get("ml_config"):
                    with st.expander("Использованные модели"):
                        st.json(result["ml_config"])

                # Показываем визуализацию
                show_cluster_visualization(result["id"])

                # Показываем детали кластеров
                if result.get("clusters"):
                    for cluster in result["clusters"]:
                        show_cluster_details(cluster)
                else:
                    st.warning("Не найдены кластеры или компании в результате.")

            elif status == "failed":
                st.markdown(
                    "<p class='status-failed'>Статус: Ошибка</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"Использовано кредитов: {result.get('credits_used', 0)}")
                st.error(f"Причина: {result.get('error', 'Неизвестная ошибка')}")
            else:
                st.markdown(
                    "<p class='status-running'>Статус: Выполняется</p>",
                    unsafe_allow_html=True,
                )
                st.info(
                    "Кластеризация в процессе. Обновите страницу через некоторое время."
                )


# Запуск приложения
if __name__ == "__main__":
    try:
        if "token" in st.session_state:
            show_main_page()
        else:
            show_login_page()
    except Exception as e:
        logger.error(f"Критическая ошибка приложения: {e}")
        st.error("Произошла критическая ошибка приложения. Перезагрузите страницу.")
