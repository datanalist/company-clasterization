# Создаем интерактивный график кластеров с названиями в 2D и 3D пространстве
import pandas as pd
import plotly.express as px
from copy import deepcopy


# Создаем DataFrame с данными для визуализации
def create_interactive_topic_plot(
    dim_model, topics, data, new_topic_names, dimensions=2
):
    """
    Создает интерактивный график кластеров топиков в 2D или 3D пространстве.

    Параметры:
    ----------
    dim_model : модель снижения размерности
        Модель, содержащая embedding_ атрибут с координатами в пространстве
    topics : array-like
        Массив с идентификаторами топиков для каждого документа
    data : DataFrame
        DataFrame с текстами документов
    new_topic_names : dict
        Словарь с названиями топиков (ключ - id топика, значение - название)
    dimensions : int, optional
        Количество измерений для визуализации (2 или 3), по умолчанию 2

    Возвращает:
    ----------
    plotly.graph_objects.Figure
        Интерактивный график кластеров
    """
    dim_embedding = deepcopy(dim_model.embedding_)
    # Проверяем, имеет ли модель уже нужное представление
    if hasattr(dim_model, "embedding_") and dim_model.embedding_.shape[1] != dimensions:
        # Если размерность не соответствует, применяем UMAP для снижения размерности
        import umap

        # Создаем и применяем UMAP для снижения размерности
        umap_model = umap.UMAP(n_components=dimensions, random_state=42)
        dim_embedding = umap_model.fit_transform(dim_embedding)

        print(
            f"Размерность снижена до {dimensions}D с помощью UMAP: {dim_embedding.shape}"
        )
    elif not hasattr(dim_model, "embedding_"):
        raise AttributeError(
            "Модель не содержит атрибут embedding_. Убедитесь, что модель правильно обучена."
        )

    # Создаем DataFrame с данными для визуализации
    plot_data = {
        "x": dim_embedding[:, 0],
        "y": dim_embedding[:, 1],
        "topic": topics,
        "text": data[
            "text"
        ],  # Добавляем текст документов для отображения при наведении
    }

    # Добавляем z-координату для 3D визуализации
    if dimensions == 3 and dim_embedding.shape[1] >= 3:
        plot_data["z"] = dim_embedding[:, 2]

    umap_data = pd.DataFrame(plot_data)

    # Добавляем названия топиков
    topic_labels = {}
    for topic_id, topic_name in new_topic_names.items():
        topic_labels[topic_id] = f"Топик {topic_id}: {topic_name}"

    # Для выбросов (topic = -1) добавляем специальную метку
    topic_labels[-1] = "Выбросы (Outliers)"

    # Создаем новый столбец с названиями топиков
    umap_data["topic_name"] = umap_data["topic"].map(
        lambda x: topic_labels.get(x, f"Топик {x}")
    )

    # Создаем интерактивный график с помощью Plotly
    if dimensions == 3:
        fig = px.scatter_3d(
            umap_data,
            x="x",
            y="y",
            z="z",
            color="topic_name",
            title="Кластеры топиков в 3D пространстве",
            labels={
                "x": "UMAP Dimension 1",
                "y": "UMAP Dimension 2",
                "z": "UMAP Dimension 3",
                "topic_name": "Название топика",
            },
            hover_data=["text"],  # Показываем текст документа при наведении
            color_discrete_sequence=px.colors.qualitative.Bold,
        )

        # Настраиваем внешний вид 3D графика
        fig.update_layout(
            legend_title_text="Топики",
            width=1200,
            height=800,
            scene=dict(
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                zaxis_title="UMAP Dimension 3",
                bgcolor="white",
            ),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
        )
    else:
        fig = px.scatter(
            umap_data,
            x="x",
            y="y",
            color="topic_name",
            title="Кластеры топиков в 2D пространстве",
            labels={
                "x": "UMAP Dimension 1",
                "y": "UMAP Dimension 2",
                "topic_name": "Название топика",
            },
            hover_data=["text"],  # Показываем текст документа при наведении
            color_discrete_sequence=px.colors.qualitative.Bold,
        )

        # Настраиваем внешний вид 2D графика
        fig.update_layout(
            legend_title_text="Топики",
            width=1200,
            height=800,
            plot_bgcolor="white",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
        )

    # Настраиваем курсор и другие параметры
    fig.update_traces(
        marker=dict(size=8, opacity=0.7),
        hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
    )

    return fig


def show_interactive_topic_plot(fig, fullscreen=True):
    """
    Отображает интерактивный график кластеров.

    Параметры:
    ----------
    fig : plotly.graph_objects.Figure
        Интерактивный график для отображения
    fullscreen : bool, optional
        Если True, отображает график в полноэкранном режиме
    """
    if fullscreen:
        fig.update_layout(height=1200, width=3000)
        fig.show(
            renderer="browser",
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "displaylogo": False,
                "responsive": True,
            },
        )
    else:
        fig.show(
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "displaylogo": False,
                "responsive": True,
            }
        )
