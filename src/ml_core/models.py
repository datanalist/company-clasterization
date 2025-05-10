from typing import Literal


class EmbeddingModel:
    """
    Класс для создания моделей эмбеддингов.

    Поддерживает различные методы создания эмбеддингов текста, включая:
    - sentence_transformer: модели Sentence Transformers
    - model2vec, huggingface, flair, spacy, use, gensim, sklearn, openai, cohere,
      multimodal, customback, customembed, tfidf: в процессе разработки

    Параметры:
    ----------
    method : str
        Метод для создания эмбеддингов
    **kwargs : dict
        Дополнительные параметры для выбранного метода

    Возвращает:
    ----------
    object
        Экземпляр модели эмбеддингов

    Вызывает:
    ---------
    ValueError
        Если указан недопустимый метод или параметры модели
    """

    def __new__(
        cls,
        method: Literal[
            "sentence_transformer",
            "model2vec",
            "huggingface",
            "flair",
            "spacy",
            "use",
            "gensim",
            "sklearn",
            "openai",
            "cohere",
            "multimodal",
            "customback",
            "customembed",
            "tfidf",
        ],
        **kwargs,
    ):
        if method == "sentence_transformer":
            from sentence_transformers import SentenceTransformer

            try:
                return SentenceTransformer(**kwargs)
            except Exception:
                raise ValueError(
                    f"Недопустимое имя модели: '{kwargs['model_name_or_path']}'"
                )
        elif method == "model2v ec":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "huggingface":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "flair":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "spacy":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "use":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "gensim":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "sklearn":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "openai":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "cohere":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "multimodal":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "customback":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "customembed":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "tfidf":
            print("In process, plese use sentence_transformer")
            return None
        else:
            raise ValueError(f"Недопустимое имя метода: '{method}'")


class ReductionModel:
    """
    Класс для создания моделей снижения размерности.

    Поддерживает различные методы снижения размерности:
    - umap: Uniform Manifold Approximation and Projection
    - svd: Singular Value Decomposition
    - pca: Principal Component Analysis
    - skip: Пропуск этапа снижения размерности
    - custom: Пользовательский метод (в настоящее время использует UMAP)

    Параметры:
    ----------
    method : str, по умолчанию "umap"
        Метод снижения размерности
    **kwargs : dict
        Дополнительные параметры для выбранного метода

    Возвращает:
    ----------
    object
        Экземпляр модели снижения размерности

    Вызывает:
    ---------
    ValueError
        Если указан недопустимый метод
    """

    def __new__(
        cls,
        method: Literal["umap", "svd", "pca", "skip", "custom"] = "umap",
        **kwargs,
    ):
        if method == "umap" or method == "custom":
            from umap import UMAP

            if method == "custom":
                print("In process, now used umap")
            return UMAP(**kwargs)
        elif method == "svd":
            from sklearn.decomposition import TruncatedSVD

            return TruncatedSVD(**kwargs)
        elif method == "pca":
            from sklearn.decomposition import PCA

            return PCA(**kwargs)
        elif method == "skip":
            from bertopic.dimensionality import BaseDimensionalityReduction

            return BaseDimensionalityReduction()
        else:
            raise ValueError(f"Недопустимый метод: {method}")


class ClusteringModel:
    """
    Класс для создания моделей кластеризации.

    Поддерживает различные алгоритмы кластеризации:
    - hdbscan: Hierarchical Density-Based Spatial Clustering of Applications with Noise
    - kmeans: K-Means кластеризация
    - agglomerative: Агломеративная иерархическая кластеризация
    - custom: Пользовательский метод (в настоящее время использует HDBSCAN)

    Параметры:
    ----------
    model_name : str, по умолчанию "hdbscan"
        Название алгоритма кластеризации
    **kwargs : dict
        Дополнительные параметры для выбранного алгоритма

    Возвращает:
    ----------
    object
        Экземпляр модели кластеризации

    Вызывает:
    ---------
    ValueError
        Если указан недопустимый алгоритм
    """

    def __new__(
        cls,
        model_name: Literal["hdbscan", "kmeans", "agglomerative", "custom"] = "hdbscan",
        **kwargs,
    ):
        if model_name == "hdbscan" or model_name == "custom":
            from sklearn.cluster import HDBSCAN

            if model_name == "custom":
                print("In process, now used hdbscan")
            return HDBSCAN(**kwargs)
        elif model_name == "kmeans":
            from sklearn.cluster import KMeans

            return KMeans(**kwargs)
        elif model_name == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering

            return AgglomerativeClustering(**kwargs)
        else:
            raise ValueError(f"Недопустимый метод: {model_name}")


class VectorizerModel:
    """
    Класс для создания моделей векторизации текста.

    Поддерживает различные методы векторизации:
    - count: CountVectorizer из sklearn
    - countonline: OnlineCountVectorizer из bertopic

    Параметры:
    ----------
    model_name : str, по умолчанию "count"
        Название метода векторизации
    **kwargs : dict
        Дополнительные параметры для выбранного метода

    Возвращает:
    ----------
    object
        Экземпляр модели векторизации

    Вызывает:
    ---------
    ValueError
        Если указан недопустимый метод
    """

    def __new__(
        cls,
        model_name: Literal["count", "countonline"] = "count",
        **kwargs,
    ):
        if model_name == "count":
            from sklearn.feature_extraction.text import CountVectorizer

            return CountVectorizer(**kwargs)
        elif model_name == "countonline":
            from bertopic.vectorizers import OnlineCountVectorizer

            return OnlineCountVectorizer(**kwargs)
        else:
            raise ValueError(f"Недопустимый метод: {model_name}")


class RepresentationModel:
    """
    Класс для создания моделей представления топиков.

    Поддерживает различные методы представления:
    - keybertinspired: Представление на основе KeyBERT
    - transformers: Генерация текста с использованием моделей Transformers
    - openai: Генерация текста с использованием API OpenAI
    - Другие методы (PartOfSpeech, MaximalMarginalRelevance, zero-shot, custom,
      gguf, llama-manual-quantization) находятся в разработке

    Параметры:
    ----------
    method : str, по умолчанию "keybertinspired"
        Метод представления топиков
    **kwargs : dict
        Дополнительные параметры для выбранного метода

    Возвращает:
    ----------
    object
        Экземпляр модели представления

    Вызывает:
    ---------
    ValueError
        Если указаны недопустимые параметры для метода transformers
    """

    def __new__(
        cls,
        method: Literal[
            "keybertinspired",
            "PartOfSpeech",
            "MaximalMarginalRelevance",
            "zero-shot",
            "custom",
            "openai",
            "transformers",
            "gguf",
            "llama-manual-quantization",
        ] = "keybertinspired",
        **kwargs,
    ):
        if method == "keybertinspired":
            from bertopic.representation import KeyBERTInspired

            return KeyBERTInspired()
        elif method == "transformers":
            from bertopic.representation import TextGeneration

            try:
                return TextGeneration(**kwargs)
            except Exception:
                raise ValueError(f"Недопустимое имя модели: '{kwargs['model_name']}'")
        elif method == "openai":
            from bertopic.representation import OpenAI

            return OpenAI(**kwargs)
        else:
            print("In process, plese use keybert or transformers")
            return None


# Создаем более осмысленные названия для топиков
# Используем KeyBERT для генерации ключевых фраз из документов каждого топика


def create_topic_names(topic_model, embedding_model, stop_words=None) -> dict:
    """
    Создает осмысленные названия для топиков с использованием KeyBERT.

    Параметры:
    ----------
    topic_model : BERTopic
        Модель BERTopic с обученными топиками
    embedding_model : модель для эмбеддингов
        Модель для создания эмбеддингов текста
    stop_words : list, optional
        Список стоп-слов для исключения из ключевых фраз

    Возвращает:
    ----------
    dict
        Словарь с новыми названиями топиков
    """
    from keybert import KeyBERT

    keybert_model = KeyBERT(model=embedding_model)

    # Получаем информацию о топиках
    topic_info = topic_model.get_topic_info()
    topic_docs = {}

    # Для каждого топика (кроме -1, который означает выбросы) получаем репрезентативные документы
    for topic_id in topic_info[topic_info["Topic"] != -1]["Topic"]:
        # Получаем документы для данного топика
        documents = topic_model.get_representative_docs(topic_id)
        topic_docs[topic_id] = " ".join(documents)

    # Создаем словарь для хранения новых названий топиков
    topic_names = {}

    # Для каждого топика генерируем ключевые фразы
    for topic_id, doc in topic_docs.items():
        # Извлекаем ключевые фразы (3 слова) из документов топика
        keywords = keybert_model.extract_keywords(
            doc, keyphrase_ngram_range=(3, 3), stop_words=stop_words, top_n=1
        )

        if keywords:
            # Берем первую ключевую фразу как название топика
            topic_names[topic_id] = keywords[0][0]
        else:
            # Если не удалось извлечь фразу, используем оригинальное название
            words = topic_model.get_topic(topic_id)
            topic_names[topic_id] = f"Топик_{topic_id}_{words[0][0]}_{words[1][0]}"

    # Переименовываем топики в модели
    topic_model.set_topic_labels(topic_names)

    return topic_names


# Применяем функцию для создания названий топиков
