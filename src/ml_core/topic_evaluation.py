"""
Модуль для оценки качества тематического моделирования.
Содержит различные метрики и методы для анализа результатов BERTopic.
"""

import numpy as np
import pandas as pd
from typing import Any
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import logging

# Импорты для CV Coherence
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import word_tokenize
import nltk
import re

# Загружаем необходимые данные для токенизации
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

logger = logging.getLogger(__name__)


class TopicModelEvaluator:
    """Класс для комплексной оценки качества тематического моделирования."""

    def __init__(
        self,
        topic_model,
        texts: list[str],
        topics: list[int],
        embeddings: np.ndarray = None,
    ):
        """
        Инициализация оценщика.

        Args:
            topic_model: Обученная модель BERTopic
            texts: Список текстов документов
            topics: Список назначенных тем для каждого документа
            embeddings: Эмбеддинги документов (опционально)
        """
        self.topic_model = topic_model
        self.texts = texts
        self.topics = topics
        self.embeddings = embeddings

    def evaluate_comprehensive(self) -> dict[str, Any]:
        """
        Комплексная оценка модели тематического моделирования.

        Returns:
            Dict с различными метриками качества
        """
        results = {}

        # Базовые метрики
        results.update(self._calculate_basic_metrics())

        # Метрики кластеризации (если есть эмбеддинги)
        if self.embeddings is not None:
            results.update(self._calculate_clustering_metrics())

        # Метрики качества тем
        results.update(self._calculate_topic_quality_metrics())

        # Метрики распределения
        results.update(self._calculate_distribution_metrics())

        logger.info("Завершена комплексная оценка модели тематического моделирования")
        return results

    def _calculate_basic_metrics(self) -> dict[str, Any]:
        """Вычисление базовых метрик."""
        metrics = {}

        # Количество тем (исключая выбросы -1)
        unique_topics = set(self.topics)
        n_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
        metrics["n_topics"] = n_topics

        # Процент выбросов
        outlier_ratio = sum(1 for t in self.topics if t == -1) / len(self.topics)
        metrics["outlier_ratio"] = outlier_ratio

        # Средний размер темы
        topic_sizes = [self.topics.count(t) for t in unique_topics if t != -1]
        metrics["avg_topic_size"] = np.mean(topic_sizes) if topic_sizes else 0
        metrics["std_topic_size"] = np.std(topic_sizes) if topic_sizes else 0

        logger.info(f"Базовые метрики: {n_topics} тем, {outlier_ratio:.2%} выбросов")
        return metrics

    def _calculate_clustering_metrics(self) -> dict[str, Any]:
        """Вычисление метрик качества кластеризации."""
        metrics = {}

        # Фильтруем выбросы для метрик кластеризации
        non_outlier_mask = np.array(self.topics) != -1
        if np.sum(non_outlier_mask) < 2:
            logger.warning("Недостаточно данных для вычисления метрик кластеризации")
            return metrics

        filtered_embeddings = self.embeddings[non_outlier_mask]
        filtered_topics = np.array(self.topics)[non_outlier_mask]

        try:
            # Silhouette Score
            if len(set(filtered_topics)) > 1:
                sil_score = silhouette_score(filtered_embeddings, filtered_topics)
                metrics["silhouette_score"] = sil_score

            # Calinski-Harabasz Index
            ch_score = calinski_harabasz_score(filtered_embeddings, filtered_topics)
            metrics["calinski_harabasz_score"] = ch_score

            # Davies-Bouldin Index
            db_score = davies_bouldin_score(filtered_embeddings, filtered_topics)
            metrics["davies_bouldin_score"] = db_score

            logger.info("Метрики кластеризации вычислены успешно")

        except Exception as e:
            logger.error(f"Ошибка при вычислении метрик кластеризации: {e}")

        return metrics

    def _calculate_topic_quality_metrics(self) -> dict[str, Any]:
        """Вычисление метрик качества тем."""
        metrics = {}

        try:
            topic_info = self.topic_model.get_topic_info()

            # Средняя длина представления темы (количество значимых слов)
            topic_representations = []
            for topic_id in topic_info["Topic"]:
                if topic_id != -1:
                    topic_words = self.topic_model.get_topic(topic_id)
                    topic_representations.append(len(topic_words))

            if topic_representations:
                metrics["avg_topic_representation_length"] = np.mean(
                    topic_representations
                )
                metrics["std_topic_representation_length"] = np.std(
                    topic_representations
                )

            # Уникальность слов в темах
            all_topic_words = []
            for topic_id in topic_info["Topic"]:
                if topic_id != -1:
                    topic_words = [
                        word for word, _ in self.topic_model.get_topic(topic_id)[:10]
                    ]
                    all_topic_words.extend(topic_words)

            if all_topic_words:
                unique_words = len(set(all_topic_words))
                total_words = len(all_topic_words)
                metrics["topic_word_uniqueness"] = (
                    unique_words / total_words if total_words > 0 else 0
                )

            logger.info("Метрики качества тем вычислены успешно")

        except Exception as e:
            logger.error(f"Ошибка при вычислении метрик качества тем: {e}")

        return metrics

    def _calculate_distribution_metrics(self) -> dict[str, Any]:
        """Вычисление метрик распределения тем."""
        metrics = {}

        try:
            # Энтропия распределения тем
            topic_counts = pd.Series(self.topics).value_counts()
            if -1 in topic_counts.index:
                topic_counts = topic_counts.drop(-1)  # Исключаем выбросы

            if len(topic_counts) > 0:
                probabilities = topic_counts / topic_counts.sum()
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                metrics["topic_entropy"] = entropy

                # Коэффициент Джини для неравномерности распределения
                gini = self._calculate_gini_coefficient(topic_counts.values)
                metrics["topic_gini_coefficient"] = gini

            logger.info("Метрики распределения вычислены успешно")

        except Exception as e:
            logger.error(f"Ошибка при вычислении метрик распределения: {e}")

        return metrics

    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Вычисление коэффициента Джини для массива значений."""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def calculate_cv_coherence(self, n_words: int = 10) -> dict[str, float]:
        """
        Вычисление CV Coherence для тем с использованием gensim.

        Args:
            n_words: Количество топовых слов для анализа

        Returns:
            Dict с CV Coherence метриками для каждой темы
        """
        coherence_scores = {}

        try:
            # Подготовка текстов для gensim
            processed_texts = self._preprocess_texts_for_coherence()

            if not processed_texts:
                logger.warning(
                    "Не удалось обработать тексты для вычисления когерентности"
                )
                return coherence_scores

            # Создаем словарь
            dictionary = Dictionary(processed_texts)

            # Получаем темы как списки слов
            topic_words = self._extract_topic_words(n_words)

            if not topic_words:
                logger.warning("Не удалось извлечь слова тем")
                return coherence_scores

            # Вычисляем CV Coherence для каждой темы
            topic_info = self.topic_model.get_topic_info()

            for i, topic_id in enumerate(topic_info["Topic"]):
                if topic_id != -1 and i < len(topic_words):
                    try:
                        # Создаем модель когерентности для одной темы
                        coherence_model = CoherenceModel(
                            topics=[topic_words[i]],
                            texts=processed_texts,
                            dictionary=dictionary,
                            coherence="c_v",
                        )

                        coherence_score = coherence_model.get_coherence()
                        coherence_scores[f"topic_{topic_id}_cv_coherence"] = (
                            coherence_score
                        )

                    except Exception as e:
                        logger.warning(
                            f"Ошибка при вычислении когерентности для темы {topic_id}: {e}"
                        )
                        coherence_scores[f"topic_{topic_id}_cv_coherence"] = 0.0

            # Средняя CV Coherence по всем темам
            individual_scores = [
                score
                for key, score in coherence_scores.items()
                if key.endswith("_cv_coherence")
            ]
            if individual_scores:
                coherence_scores["avg_cv_coherence"] = np.mean(individual_scores)

            logger.info(f"CV Coherence вычислена для {len(individual_scores)} тем")

        except Exception as e:
            logger.error(f"Ошибка при вычислении CV Coherence: {e}")

        return coherence_scores

    def _preprocess_texts_for_coherence(self) -> list[list[str]]:
        """
        Предобработка текстов для вычисления CV Coherence.

        Returns:
            Список токенизированных текстов
        """
        processed_texts = []

        try:
            for text in self.texts:
                # Очистка текста
                clean_text = re.sub(r"[^\w\s]", " ", text.lower())
                clean_text = re.sub(r"\s+", " ", clean_text).strip()

                # Токенизация
                tokens = word_tokenize(clean_text, language="russian")

                # Фильтрация токенов (минимальная длина, исключение цифр)
                filtered_tokens = [
                    token for token in tokens if len(token) > 2 and token.isalpha()
                ]

                if filtered_tokens:
                    processed_texts.append(filtered_tokens)

            logger.info(
                f"Обработано {len(processed_texts)} текстов для вычисления когерентности"
            )

        except Exception as e:
            logger.error(f"Ошибка при предобработке текстов: {e}")

        return processed_texts

    def _extract_topic_words(self, n_words: int) -> list[list[str]]:
        """
        Извлечение слов тем для вычисления когерентности.

        Args:
            n_words: Количество топовых слов для каждой темы

        Returns:
            Список списков слов для каждой темы
        """
        topic_words = []

        try:
            topic_info = self.topic_model.get_topic_info()

            for topic_id in topic_info["Topic"]:
                if topic_id != -1:
                    # Получаем топовые слова темы
                    words = [
                        word
                        for word, _ in self.topic_model.get_topic(topic_id)[:n_words]
                    ]
                    topic_words.append(words)

            logger.info(f"Извлечено слов для {len(topic_words)} тем")

        except Exception as e:
            logger.error(f"Ошибка при извлечении слов тем: {e}")

        return topic_words

    def generate_evaluation_report(self, save_path: str = None) -> pd.DataFrame:
        """
        Генерация отчета с результатами оценки.

        Args:
            save_path: Путь для сохранения отчета (опционально)

        Returns:
            DataFrame с результатами оценки
        """
        # Получаем все метрики
        comprehensive_results = self.evaluate_comprehensive()
        coherence_results = self.calculate_cv_coherence()

        # Объединяем результаты
        all_results = {**comprehensive_results, **coherence_results}

        # Создаем DataFrame
        report_df = pd.DataFrame([all_results]).T
        report_df.columns = ["Value"]
        report_df.index.name = "Metric"

        # Добавляем интерпретацию
        report_df["Interpretation"] = report_df.index.map(
            self._get_metric_interpretation
        )

        if save_path:
            report_df.to_csv(save_path)
            logger.info(f"Отчет об оценке сохранен в {save_path}")

        return report_df

    def _get_metric_interpretation(self, metric_name: str) -> str:
        """Получение интерпретации для метрики."""
        interpretations = {
            "n_topics": "Количество обнаруженных тем",
            "outlier_ratio": "Доля документов-выбросов (чем меньше, тем лучше)",
            "silhouette_score": "Качество кластеризации (-1 до 1, выше лучше)",
            "calinski_harabasz_score": "Отношение межгрупповой/внутригрупповой дисперсии (выше лучше)",
            "davies_bouldin_score": "Средняя схожесть кластеров (ниже лучше)",
            "topic_entropy": "Энтропия распределения тем (выше = более равномерно)",
            "topic_gini_coefficient": "Неравномерность распределения (0-1, ниже лучше)",
            "avg_cv_coherence": "Средняя CV Coherence тем (выше лучше)",
            "topic_word_uniqueness": "Уникальность слов в темах (выше лучше)",
        }

        return interpretations.get(metric_name, "Специфическая метрика")


def quick_evaluate_bertopic(
    topic_model, texts: list[str], topics: list[int], embeddings: np.ndarray = None
) -> dict[str, Any]:
    """
    Быстрая оценка модели BERTopic с ключевыми метриками.

    Args:
        topic_model: Обученная модель BERTopic
        texts: Тексты документов
        topics: Назначенные темы
        embeddings: Эмбеддинги документов (опционально)

    Returns:
        Словарь с ключевыми метриками
    """
    evaluator = TopicModelEvaluator(topic_model, texts, topics, embeddings)

    # Получаем только ключевые метрики
    basic_metrics = evaluator._calculate_basic_metrics()
    coherence_metrics = evaluator.calculate_cv_coherence()

    key_metrics = {
        "num_topics": basic_metrics.get("n_topics", 0),
        "outlier_percentage": basic_metrics.get("outlier_ratio", 0) * 100,
        "avg_topic_size": basic_metrics.get("avg_topic_size", 0),
        "avg_cv_coherence": coherence_metrics.get("avg_cv_coherence", 0),
    }

    if embeddings is not None:
        clustering_metrics = evaluator._calculate_clustering_metrics()
        key_metrics["silhouette_score"] = clustering_metrics.get("silhouette_score", 0)

    logger.info("Быстрая оценка модели завершена")
    return key_metrics
