from typing import Literal
import logging

# Настройка логирования
logger = logging.getLogger(__name__)


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
            except Exception as e:
                logger.error(f"Ошибка при создании модели SentenceTransformer: {e}")
                raise ValueError(
                    f"Недопустимое имя модели или параметры: '{kwargs.get('model_name_or_path', 'не указано')}'"
                )
        elif method == "model2vec":
            logger.warning(
                "model2vec в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "huggingface":
            logger.warning(
                "huggingface в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "flair":
            logger.warning(
                "flair в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "spacy":
            logger.warning(
                "spacy в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "use":
            logger.warning(
                "use в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "gensim":
            logger.warning(
                "gensim в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "sklearn":
            logger.warning(
                "sklearn в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "openai":
            logger.warning(
                "openai в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "cohere":
            logger.warning(
                "cohere в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "multimodal":
            logger.warning(
                "multimodal в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "customback":
            logger.warning(
                "customback в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "customembed":
            logger.warning(
                "customembed в процессе разработки, используйте sentence_transformer"
            )
            return None
        elif method == "tfidf":
            logger.warning(
                "tfidf в процессе разработки, используйте sentence_transformer"
            )
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
        try:
            if method == "umap" or method == "custom":
                from umap import UMAP

                if method == "custom":
                    logger.warning("custom в процессе разработки, используется umap")
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
        except ImportError as e:
            logger.error(f"Ошибка импорта для метода {method}: {e}")
            raise ValueError(
                f"Не удалось импортировать зависимости для метода '{method}': {e}"
            )
        except Exception as e:
            logger.error(
                f"Ошибка при создании модели снижения размерности {method}: {e}"
            )
            raise ValueError(f"Ошибка при создании модели снижения размерности: {e}")


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
    method : str, по умолчанию "hdbscan"
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
        method: Literal["hdbscan", "kmeans", "agglomerative", "custom"] = "hdbscan",
        **kwargs,
    ):
        try:
            if method == "hdbscan" or method == "custom":
                from sklearn.cluster import HDBSCAN

                if method == "custom":
                    logger.warning("custom в процессе разработки, используется hdbscan")
                return HDBSCAN(**kwargs)
            elif method == "kmeans":
                from sklearn.cluster import KMeans

                return KMeans(**kwargs)
            elif method == "agglomerative":
                from sklearn.cluster import AgglomerativeClustering

                return AgglomerativeClustering(**kwargs)
            else:
                raise ValueError(f"Недопустимый метод: {method}")
        except ImportError as e:
            logger.error(f"Ошибка импорта для метода {method}: {e}")
            raise ValueError(
                f"Не удалось импортировать зависимости для метода '{method}': {e}"
            )
        except Exception as e:
            logger.error(f"Ошибка при создании модели кластеризации {method}: {e}")
            raise ValueError(f"Ошибка при создании модели кластеризации: {e}")


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
        try:
            if model_name == "count":
                from sklearn.feature_extraction.text import CountVectorizer

                return CountVectorizer(**kwargs)
            elif model_name == "countonline":
                from bertopic.vectorizers import OnlineCountVectorizer

                return OnlineCountVectorizer(**kwargs)
            else:
                raise ValueError(f"Недопустимый метод: {model_name}")
        except ImportError as e:
            logger.error(f"Ошибка импорта для метода {model_name}: {e}")
            raise ValueError(
                f"Не удалось импортировать зависимости для метода '{model_name}': {e}"
            )
        except Exception as e:
            logger.error(f"Ошибка при создании модели векторизации {model_name}: {e}")
            raise ValueError(f"Ошибка при создании модели векторизации: {e}")


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
        try:
            if method == "keybertinspired":
                from bertopic.representation import KeyBERTInspired

                return KeyBERTInspired()
            elif method == "transformers":
                from bertopic.representation import TextGeneration

                try:
                    return TextGeneration(**kwargs)
                except Exception as e:
                    logger.error(f"Ошибка при создании TextGeneration: {e}")
                    raise ValueError(
                        f"Недопустимое имя модели или параметры: '{kwargs.get('model', 'не указано')}'"
                    )
            elif method == "openai":
                from bertopic.representation import OpenAI

                return OpenAI(**kwargs)
            else:
                logger.warning(
                    f"Метод {method} в процессе разработки, используйте keybertinspired или transformers"
                )
                return None
        except ImportError as e:
            logger.error(f"Ошибка импорта для метода {method}: {e}")
            raise ValueError(
                f"Не удалось импортировать зависимости для метода '{method}': {e}"
            )
        except Exception as e:
            logger.error(f"Ошибка при создании модели представления {method}: {e}")
            raise ValueError(f"Ошибка при создании модели представления: {e}")


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
    try:
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
            try:
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
                    if words and len(words) >= 2:
                        topic_names[topic_id] = (
                            f"Топик_{topic_id}_{words[0][0]}_{words[1][0]}"
                        )
                    else:
                        topic_names[topic_id] = f"Топик_{topic_id}"
            except Exception as e:
                logger.warning(
                    f"Ошибка при создании названия для топика {topic_id}: {e}"
                )
                topic_names[topic_id] = f"Топик_{topic_id}"

        # Переименовываем топики в модели
        try:
            topic_model.set_topic_labels(topic_names)
        except Exception as e:
            logger.warning(f"Не удалось установить метки топиков в модели: {e}")

        return topic_names

    except ImportError as e:
        logger.error(f"Ошибка импорта KeyBERT: {e}")
        # Возвращаем дефолтные названия
        topic_info = topic_model.get_topic_info()
        return {
            topic_id: f"Топик_{topic_id}"
            for topic_id in topic_info[topic_info["Topic"] != -1]["Topic"]
        }
    except Exception as e:
        logger.error(f"Ошибка при создании названий топиков: {e}")
        # Возвращаем дефолтные названия
        try:
            topic_info = topic_model.get_topic_info()
            return {
                topic_id: f"Топик_{topic_id}"
                for topic_id in topic_info[topic_info["Topic"] != -1]["Topic"]
            }
        except Exception:
            return {}


# Применяем функцию для создания названий топиков
