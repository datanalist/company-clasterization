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
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        texts = del_na_news['text'].tolist()
        context.log.info("Начинаем создание эмбеддингов...")
        embeddings = model.encode(texts)
        df_with_embeddings = del_na_news.copy()
        df_with_embeddings['embedding'] = list(embeddings)
        context.log.info(f"Созданы эмбеддинги для {len(df_with_embeddings)} новостей")
        return Output(value=df_with_embeddings)
    except Exception as e:
        context.log.error(f"Ошибка при создании эмбеддингов: {e}")
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
        embeddings_array = np.array(get_embeddings['embedding'].tolist())
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        get_embeddings['cluster'] = kmeans.fit_predict(embeddings_array)
        return Output(value=get_embeddings)
    except Exception as e:
        context.log.error(f"Ошибка при кластеризации: {e}")
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
        embeddings_array = np.array(cluster_news['embedding'].tolist())
        tsne = TSNE(n_components=3, random_state=42, perplexity=2)
        tsne_result = tsne.fit_transform(embeddings_array)
        cluster_news['tsne_x'] = tsne_result[:, 0]
        cluster_news['tsne_y'] = tsne_result[:, 1]
        cluster_news['tsne_z'] = tsne_result[:, 2]
        fig = px.scatter_3d(cluster_news, x='tsne_x', y='tsne_y', z='tsne_z', color='cluster', hover_data=['title'])
        fig.show()
        return Output(value=cluster_news)
    except Exception as e:
        context.log.error(f"Ошибка при визуализации: {e}")
        return Output(value=cluster_news)
