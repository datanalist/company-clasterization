import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from dagster import asset, Output
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px

from .parse_lenta import parse_lenta_news

@asset
def lenta_news():
    """
    Получение новостей с сайта Lenta.ru.

    Этот ассет использует функцию parse_lenta_news для сбора новостей
    с сайта Lenta.ru и возвращает их в виде DataFrame.

    Returns:
        DataFrame с новостями, содержащий заголовки, тексты, URL и другие данные
    """
    df = parse_lenta_news()
    return df

@asset
def del_na_news(context, lenta_news: pd.DataFrame):
    """
    Удаление новостей с пропущенными значениями в поле 'text'.

    Args:
        context: Контекст Dagster
        lenta_news: DataFrame с новостями

    Returns:
        DataFrame без строк с пропущенными значениями в поле 'text'
    """
    df = lenta_news.dropna(subset=["text"])
    context.log.info(df)
    return Output(value=df)

@asset
def get_embeddings(context, del_na_news: pd.DataFrame):
    """
    Получение эмбеддингов из текстов новостей.

    Args:
        context: Контекст Dagster
        del_na_news: DataFrame с новостями без пропущенных значений

    Returns:
        DataFrame с добавленными эмбеддингами
    """
    try:


        # Загрузка модели для создания эмбеддингов
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

        # Получение текстов из DataFrame
        texts = del_na_news['text'].tolist()

        # Создание эмбеддингов
        context.log.info("Начинаем создание эмбеддингов...")
        embeddings = model.encode(texts)

        # Добавление эмбеддингов в DataFrame
        df_with_embeddings = del_na_news.copy()
        df_with_embeddings['embedding'] = list(embeddings)

        context.log.info(f"Созданы эмбеддинги для {len(df_with_embeddings)} новостей")

        return Output(value=df_with_embeddings)

    except Exception as e:
        context.log.error(f"Ошибка при создании эмбеддингов: {e}")
        # Возвращаем исходный DataFrame в случае ошибки
        return Output(value=del_na_news)


@asset
def cluster_news(context, get_embeddings: pd.DataFrame):
    """
    Кластеризация новостей на основе эмбеддингов.

    Args:
        context: Контекст Dagster
        get_embeddings: DataFrame с новостями и эмбеддингами

    Returns:
        DataFrame с кластерами
    """
    try:
        # Преобразование списка эмбеддингов в numpy массив
        embeddings_array = np.array(get_embeddings['embedding'].tolist())

        # Определение оптимального количества кластеров (можно изменить)
        n_clusters = 5

        # Применение KMeans для кластеризации
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        get_embeddings['cluster'] = kmeans.fit_predict(embeddings_array)

        return Output(value=get_embeddings)

    except Exception as e:
        context.log.error(f"Ошибка при кластеризации: {e}")
        # Возвращаем исходный DataFrame в случае ошибки
        return Output(value=get_embeddings)


@asset
def plot_news(context, cluster_news: pd.DataFrame):
    """
    Визуализация кластеризации новостей.

    Args:
        context: Контекст Dagster
        cluster_news: DataFrame с новостями и кластерами

    Returns:
        Визуализация кластеризации
    """
    try:
        # Преобразование списка эмбеддингов в numpy массив
        embeddings_array = np.array(cluster_news['embedding'].tolist())

        # Применение t-SNE для визуализации кластеров
        tsne = TSNE(n_components=2, random_state=42, perplexity=1)
        tsne_result = tsne.fit_transform(embeddings_array)

        # Добавление результатов t-SNE в DataFrame
        cluster_news['tsne_x'] = tsne_result[:, 0]
        cluster_news['tsne_y'] = tsne_result[:, 1]

        # Создание графика
        fig = px.scatter(cluster_news, x='tsne_x', y='tsne_y', color='cluster', hover_data=['title'])

        # Отображение графика
        fig.show()

        return Output(value=cluster_news)

    except Exception as e:
        context.log.error(f"Ошибка при визуализации: {e}")
        # Возвращаем исходный DataFrame в случае ошибки
        return Output(value=cluster_news)
