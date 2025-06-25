"""
Пример использования метрик для оценки качества тематического моделирования.
Демонстрирует различные подходы к оценке BERTopic модели.
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from src.ml_core.topic_evaluation import TopicModelEvaluator, quick_evaluate_bertopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_bertopic_model_comprehensive():
    """
    Комплексная оценка BERTopic модели с использованием всех доступных метрик.
    """
    print("🔍 Запуск комплексной оценки BERTopic модели...")

    # Загружаем данные (замените на ваш путь)
    data_path = "../data/processed/merged_news_data.csv"

    try:
        df = pd.read_csv(data_path)
        texts = df['text'].tolist()
        logger.info(f"Загружено {len(texts)} документов")
    except FileNotFoundError:
        logger.error(f"Файл {data_path} не найден")
        return

    # Инициализируем и обучаем модель (пример)
    print("🚀 Инициализация и обучение BERTopic модели...")

    # Загружаем модель эмбеддингов
    embedding_model = SentenceTransformer("cointegrated/rubert-tiny")

    # Создаем модель BERTopic
    topic_model = BERTopic(
        language="russian",
        embedding_model=embedding_model,
        min_topic_size=5,
        nr_topics="auto"
    )

    # Обучаем модель
    topics, probs = topic_model.fit_transform(texts[:100])  # Используем первые 100 для примера

    # Получаем эмбеддинги
    embeddings = embedding_model.encode(texts[:100], show_progress_bar=True)

    print("📊 Выполнение комплексной оценки...")

    # Создаем оценщик
    evaluator = TopicModelEvaluator(
        topic_model=topic_model,
        texts=texts[:100],
        topics=topics,
        embeddings=embeddings
    )

    # Выполняем комплексную оценку
    results = evaluator.evaluate_comprehensive()

    # Генерируем отчет
    report = evaluator.generate_evaluation_report(
        save_path="../data/interim/topic_evaluation_report.csv"
    )

    print("\n📈 Результаты оценки:")
    print("=" * 50)

    for metric, info in report.iterrows():
        print(f"{metric:30s}: {info['Value']:10.4f} - {info['Interpretation']}")

    # Дополнительная оценка CV Coherence
    coherence_results = evaluator.calculate_cv_coherence()
    print(f"\n🎯 Средняя CV Coherence тем: {coherence_results.get('avg_cv_coherence', 0):.4f}")

    return report


def quick_evaluation_example():
    """
    Пример быстрой оценки модели для получения ключевых метрик.
    """
    print("\n⚡ Быстрая оценка модели...")

    # Загружаем данные
    data_path = "../data/processed/merged_news_data.csv"

    try:
        df = pd.read_csv(data_path)
        texts = df['text'].tolist()[:50]  # Используем меньше данных для примера
    except FileNotFoundError:
        logger.error(f"Файл {data_path} не найден")
        return

    # Простая модель для демонстрации
    embedding_model = SentenceTransformer("cointegrated/rubert-tiny")
    topic_model = BERTopic(language="russian", min_topic_size=3)

    topics, probs = topic_model.fit_transform(texts)
    embeddings = embedding_model.encode(texts)

    # Быстрая оценка
    quick_results = quick_evaluate_bertopic(
        topic_model=topic_model,
        texts=texts,
        topics=topics,
        embeddings=embeddings
    )

    print("\n🔢 Ключевые метрики:")
    print("-" * 30)
    for metric, value in quick_results.items():
        print(f"{metric:20s}: {value:.4f}")


def compare_models_example():
    """
    Пример сравнения нескольких моделей тематического моделирования.
    """
    print("\n🔄 Сравнение различных моделей...")

    # Загружаем данные
    data_path = "../data/processed/merged_news_data.csv"

    try:
        df = pd.read_csv(data_path)
        texts = df['text'].tolist()[:30]  # Небольшой набор для демонстрации
    except FileNotFoundError:
        logger.error(f"Файл {data_path} не найден")
        return

    # Модели для сравнения
    models_config = [
        {
            'name': 'BERTopic_min5',
            'model': BERTopic(language="russian", min_topic_size=5)
        },
        {
            'name': 'BERTopic_min3',
            'model': BERTopic(language="russian", min_topic_size=3)
        }
    ]

    comparison_results = []

    for config in models_config:
        print(f"Оценка модели: {config['name']}")

        # Обучаем модель
        topics, probs = config['model'].fit_transform(texts)

        # Быстрая оценка
        quick_results = quick_evaluate_bertopic(
            topic_model=config['model'],
            texts=texts,
            topics=topics
        )

        quick_results['model_name'] = config['name']
        comparison_results.append(quick_results)

    # Создаем DataFrame для сравнения
    comparison_df = pd.DataFrame(comparison_results)

    print("\n📊 Сравнение моделей:")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # Сохраняем результаты
    comparison_df.to_csv("../data/interim/models_comparison.csv", index=False)
    print("\n💾 Результаты сравнения сохранены в models_comparison.csv")


def topic_stability_analysis():
    """
    Анализ стабильности тем при множественных запусках.
    """
    print("\n🔄 Анализ стабильности тем...")

    # Загружаем данные
    data_path = "../data/processed/merged_news_data.csv"

    try:
        df = pd.read_csv(data_path)
        texts = df['text'].tolist()[:40]
    except FileNotFoundError:
        logger.error(f"Файл {data_path} не найден")
        return

    n_runs = 3
    stability_results = []

    for run in range(n_runs):
        print(f"Запуск {run + 1}/{n_runs}")

        # Создаем модель с разными random_state
        topic_model = BERTopic(
            language="russian",
            min_topic_size=3,
            random_state=run
        )

        topics, probs = topic_model.fit_transform(texts)

        # Получаем основные метрики
        quick_results = quick_evaluate_bertopic(
            topic_model=topic_model,
            texts=texts,
            topics=topics
        )

        quick_results['run'] = run + 1
        stability_results.append(quick_results)

    # Анализируем стабильность
    stability_df = pd.DataFrame(stability_results)

    print("\n📈 Анализ стабильности:")
    print("=" * 50)

    # Вычисляем статистики
    numeric_cols = ['num_topics', 'outlier_percentage', 'avg_topic_size', 'avg_cv_coherence']

    for col in numeric_cols:
        if col in stability_df.columns:
            mean_val = stability_df[col].mean()
            std_val = stability_df[col].std()
            print(f"{col:20s}: μ={mean_val:.3f}, σ={std_val:.3f}")

    # Сохраняем результаты
    stability_df.to_csv("../data/interim/stability_analysis.csv", index=False)
    print("\n💾 Результаты анализа стабильности сохранены")


if __name__ == "__main__":
    """
    Основная функция для запуска примеров оценки.
    """

    print("🚀 Запуск примеров оценки качества тематического моделирования")
    print("=" * 70)

    try:
        # 1. Комплексная оценка
        comprehensive_report = evaluate_bertopic_model_comprehensive()

        # 2. Быстрая оценка
        quick_evaluation_example()

        # 3. Сравнение моделей
        compare_models_example()

        # 4. Анализ стабильности
        topic_stability_analysis()

        print("\n✅ Все примеры выполнены успешно!")
        print("📁 Результаты сохранены в папке data/interim/")

    except Exception as e:
        logger.error(f"Ошибка при выполнении примеров: {e}")
        print(f"\n❌ Произошла ошибка: {e}")


# Дополнительные функции для специфических метрик

def calculate_topic_diversity(topic_model, n_words=10):
    """
    Вычисляет разнообразие тем на основе уникальности слов.

    Args:
        topic_model: Обученная BERTopic модель
        n_words: Количество топовых слов для анализа

    Returns:
        float: Коэффициент разнообразия тем (0-1)
    """
    topic_info = topic_model.get_topic_info()
    all_words = set()
    topic_words_count = 0

    for topic_id in topic_info['Topic']:
        if topic_id != -1:
            words = [word for word, _ in topic_model.get_topic(topic_id)[:n_words]]
            all_words.update(words)
            topic_words_count += len(words)

    # Разнообразие = уникальные слова / общее количество слов
    diversity = len(all_words) / topic_words_count if topic_words_count > 0 else 0
    return diversity


def calculate_topic_coverage(topic_model, texts, topics):
    """
    Вычисляет покрытие документов темами (процент документов в темах vs выбросы).

    Args:
        topic_model: Обученная BERTopic модель
        texts: Список текстов
        topics: Список назначенных тем

    Returns:
        dict: Метрики покрытия
    """
    total_docs = len(topics)
    outliers = sum(1 for t in topics if t == -1)
    covered_docs = total_docs - outliers

    coverage_metrics = {
        'coverage_ratio': covered_docs / total_docs,
        'outlier_ratio': outliers / total_docs,
        'total_documents': total_docs,
        'covered_documents': covered_docs,
        'outlier_documents': outliers
    }

    return coverage_metrics


def print_evaluation_summary(evaluation_results):
    """
    Выводит красивую сводку результатов оценки.

    Args:
        evaluation_results: Результаты оценки из TopicModelEvaluator
    """
    print("\n" + "="*60)
    print("📋 СВОДКА РЕЗУЛЬТАТОВ ОЦЕНКИ ТЕМАТИЧЕСКОГО МОДЕЛИРОВАНИЯ")
    print("="*60)

    # Основные метрики
    print("\n🔢 ОСНОВНЫЕ ПОКАЗАТЕЛИ:")
    print("-" * 40)
    print(f"Количество тем: {evaluation_results.get('n_topics', 'N/A')}")
    print(f"Процент выбросов: {evaluation_results.get('outlier_ratio', 0)*100:.1f}%")
    print(f"Средний размер темы: {evaluation_results.get('avg_topic_size', 'N/A'):.1f}")

    # Качество кластеризации
    if 'silhouette_score' in evaluation_results:
        print(f"\n🎯 КАЧЕСТВО КЛАСТЕРИЗАЦИИ:")
        print("-" * 40)
        print(f"Silhouette Score: {evaluation_results['silhouette_score']:.3f}")
        print(f"Calinski-Harabasz: {evaluation_results.get('calinski_harabasz_score', 'N/A'):.1f}")
        print(f"Davies-Bouldin: {evaluation_results.get('davies_bouldin_score', 'N/A'):.3f}")

    # Распределение тем
    if 'topic_entropy' in evaluation_results:
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ТЕМ:")
        print("-" * 40)
        print(f"Энтропия распределения: {evaluation_results['topic_entropy']:.3f}")
        print(f"Коэффициент Джини: {evaluation_results.get('topic_gini_coefficient', 'N/A'):.3f}")

    print("\n" + "="*60)
