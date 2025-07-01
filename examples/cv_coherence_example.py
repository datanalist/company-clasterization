"""
Пример использования CV Coherence для оценки качества BERTopic модели.
Демонстрирует как использовать gensim CoherenceModel для вычисления CV Coherence.
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


def demonstrate_cv_coherence():
    """
    Демонстрация использования CV Coherence для оценки BERTopic модели.
    """
    print("🎯 Демонстрация CV Coherence для BERTopic")
    print("=" * 50)

    # Загружаем данные
    try:
        data_path = "../data/processed/merged_news_data.csv"
        df = pd.read_csv(data_path)
        texts = df['text'].tolist()[:50]  # Используем первые 50 для демонстрации
        logger.info(f"Загружено {len(texts)} документов")
    except FileNotFoundError:
        logger.error(f"Файл не найден. Используем пример данных.")
        # Создаем пример данных для демонстрации
        texts = [
            "Президент России встретился с премьер-министром для обсуждения экономических вопросов",
            "Спортсмены готовятся к олимпийским играм и тренируются каждый день",
            "Новые технологии искусственного интеллекта меняют нашу жизнь",
            "Врачи рекомендуют здоровое питание и регулярные физические упражнения",
            "Климатические изменения влияют на экологию планеты"
        ] * 10  # Дублируем для достаточного количества данных

    print(f"📄 Количество документов: {len(texts)}")

    # Создаем и обучаем BERTopic модель
    print("\n🚀 Создание и обучение BERTopic модели...")

    # Используем легкую модель для демонстрации
    embedding_model = SentenceTransformer("cointegrated/rubert-tiny")
    topic_model = BERTopic(
        language="russian",
        embedding_model=embedding_model,
        min_topic_size=3,
        nr_topics="auto"
    )

    # Обучаем модель
    topics, probs = topic_model.fit_transform(texts)

    print(f"✅ Модель обучена. Найдено тем: {len(set(topics)) - (1 if -1 in topics else 0)}")

    # Получаем эмбеддинги для дополнительных метрик
    print("\n🔢 Получение эмбеддингов...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # Создаем оценщик
    print("\n📊 Вычисление CV Coherence...")
    evaluator = TopicModelEvaluator(
        topic_model=topic_model,
        texts=texts,
        topics=topics,
        embeddings=embeddings
    )

    # Вычисляем CV Coherence
    cv_coherence_results = evaluator.calculate_cv_coherence(n_words=10)

    print("\n🎯 Результаты CV Coherence:")
    print("-" * 40)

    for metric, value in cv_coherence_results.items():
        if isinstance(value, float):
            print(f"{metric:25s}: {value:.4f}")

    # Быстрая оценка с CV Coherence
    print("\n⚡ Быстрая оценка модели:")
    print("-" * 40)

    quick_results = quick_evaluate_bertopic(
        topic_model=topic_model,
        texts=texts,
        topics=topics,
        embeddings=embeddings
    )

    for metric, value in quick_results.items():
        if isinstance(value, (int, float)):
            print(f"{metric:20s}: {value:.4f}")

    # Комплексная оценка
    print("\n📈 Комплексная оценка модели:")
    print("-" * 40)

    comprehensive_results = evaluator.evaluate_comprehensive()

    # Выводим ключевые метрики
    key_metrics = [
        'n_topics', 'outlier_ratio', 'silhouette_score',
        'topic_entropy', 'topic_word_uniqueness'
    ]

    for metric in key_metrics:
        if metric in comprehensive_results:
            value = comprehensive_results[metric]
            if isinstance(value, float):
                print(f"{metric:20s}: {value:.4f}")
            else:
                print(f"{metric:20s}: {value}")

    # Интерпретация результатов
    print("\n💡 Интерпретация результатов:")
    print("-" * 40)

    avg_cv_coherence = cv_coherence_results.get('avg_cv_coherence', 0)

    if avg_cv_coherence > 0.5:
        coherence_quality = "Отличная"
    elif avg_cv_coherence > 0.3:
        coherence_quality = "Хорошая"
    elif avg_cv_coherence > 0.1:
        coherence_quality = "Удовлетворительная"
    else:
        coherence_quality = "Низкая"

    print(f"CV Coherence качество: {coherence_quality} ({avg_cv_coherence:.4f})")

    outlier_ratio = comprehensive_results.get('outlier_ratio', 0)
    if outlier_ratio < 0.1:
        coverage_quality = "Отличное"
    elif outlier_ratio < 0.3:
        coverage_quality = "Хорошее"
    else:
        coverage_quality = "Низкое"

    print(f"Покрытие данных: {coverage_quality} ({(1-outlier_ratio)*100:.1f}%)")

    # Показываем топ-слова для каждой темы
    print("\n🏷️ Топ-слова для каждой темы:")
    print("-" * 40)

    topic_info = topic_model.get_topic_info()
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)[:5]  # Топ 5 слов
            words_str = ", ".join([word for word, _ in topic_words])
            print(f"Тема {topic_id:2d}: {words_str}")

    return evaluator, cv_coherence_results


def compare_cv_coherence_parameters():
    """
    Сравнение CV Coherence с разным количеством слов.
    """
    print("\n🔄 Сравнение CV Coherence с разным количеством слов")
    print("=" * 60)

    # Простые тестовые данные
    texts = [
        "Президент России встретился с министрами правительства",
        "Спортсмены тренируются к олимпийским соревнованиям",
        "Технологии искусственного интеллекта развиваются быстро",
        "Врачи рекомендуют здоровый образ жизни",
        "Экология планеты требует нашего внимания"
    ] * 8

    # Создаем модель
    embedding_model = SentenceTransformer("cointegrated/rubert-tiny")
    topic_model = BERTopic(
        language="russian",
        embedding_model=embedding_model,
        min_topic_size=3
    )

    topics, _ = topic_model.fit_transform(texts)

    # Создаем оценщик
    evaluator = TopicModelEvaluator(topic_model, texts, topics)

    # Тестируем разное количество слов
    word_counts = [5, 10, 15, 20]
    results = []

    for n_words in word_counts:
        print(f"Тестирование с {n_words} словами...")
        coherence_results = evaluator.calculate_cv_coherence(n_words=n_words)
        avg_coherence = coherence_results.get('avg_cv_coherence', 0)
        results.append({
            'n_words': n_words,
            'avg_cv_coherence': avg_coherence
        })

    # Выводим результаты
    print("\n📊 Результаты сравнения:")
    print("-" * 30)
    for result in results:
        print(f"{result['n_words']:2d} слов: {result['avg_cv_coherence']:.4f}")

    return results


if __name__ == "__main__":
    """
    Основная функция для демонстрации CV Coherence.
    """

    print("🎯 CV Coherence для BERTopic - Демонстрация")
    print("=" * 60)

    try:
        # Основная демонстрация
        evaluator, results = demonstrate_cv_coherence()

        # Сравнение параметров
        comparison_results = compare_cv_coherence_parameters()

        print("\n✅ Демонстрация завершена успешно!")
        print("\n📚 Информация о CV Coherence:")
        print("-" * 40)
        print("• CV Coherence измеряет семантическую связность слов в темах")
        print("• Диапазон: 0 до 1 (выше лучше)")
        print("• Хорошие значения: > 0.3")
        print("• Отличные значения: > 0.5")
        print("• Использует векторные представления слов для более точной оценки")

    except Exception as e:
        logger.error(f"Ошибка при демонстрации: {e}")
        print(f"\n❌ Произошла ошибка: {e}")
