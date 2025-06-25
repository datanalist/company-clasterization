# 📊 Руководство по Оценке Качества Тематического Моделирования

## Обзор

Данное руководство содержит исчерпывающий перечень метрик и мер для оценки качества моделей тематического моделирования (Topic Modeling), особенно для BERTopic. Включает как количественные, так и качественные подходы к оценке.

## 🎯 Категории Метрик

### 1. 📈 Количественные Метрики

#### A. Метрики Когерентности (Coherence Metrics)

**Описание**: Измеряют семантическую связность слов внутри тем.

- **UMass Coherence**
  - Диапазон: -∞ до +∞ (выше лучше)
  - Формула: Основана на совместной встречаемости слов
  - Интерпретация: Положительные значения указывают на хорошую когерентность

- **CV Coherence**
  - Диапазон: 0 до 1 (выше лучше)
  - Особенности: Более точная, учитывает контекст
  - Рекомендация: Предпочтительная метрика для большинства случаев

- **NPMI (Normalized Pointwise Mutual Information)**
  - Диапазон: -1 до +1 (выше лучше)
  - Преимущества: Нормализованная, легко интерпретируемая

#### B. Метрики Кластеризации

**Silhouette Score**
- Диапазон: -1 до +1
- Интерпретация:
  - > 0.5 - отличная кластеризация
  - 0.2-0.5 - разумная кластеризация
  - < 0.2 - слабая кластеризация

**Calinski-Harabasz Index**
- Диапазон: 0 до +∞ (выше лучше)
- Измеряет: Отношение межгрупповой к внутригрупповой дисперсии

**Davies-Bouldin Index**
- Диапазон: 0 до +∞ (ниже лучше)
- Измеряет: Среднюю схожесть между кластерами

#### C. Метрики Модели

**Perplexity**
- Применимость: В основном для LDA
- Интерпретация: Ниже лучше
- Ограничения: Не всегда коррелирует с качеством тем

**Log-likelihood**
- Измеряет: Вероятность данных при модели
- Использование: Для сравнения моделей

### 2. 🎨 Качественные Метрики

#### A. Интерпретируемость Тем (Topic Interpretability)

**Человеческая Оценка**
- Шкала: 1-5 или 1-10
- Критерии:
  - Семантическая связность слов
  - Понятность темы
  - Различимость тем

**Word Intrusion Task**
- Метод: Введение постороннего слова в тему
- Оценка: Способность экспертов найти лишнее слово

**Topic Intrusion Task**
- Метод: Добавление посторонней темы к документу
- Оценка: Способность экспертов найти лишнюю тему

#### B. Качество Меток Тем

**Автоматические Метки**
- Методы: KeyBERT, TextRank, LLM-генерация
- Оценка: Соответствие содержанию темы

**Экспертная Оценка Меток**
- Критерии: Точность, понятность, краткость

### 3. 🔄 Метрики Стабильности

#### A. Topic Stability

**Jaccard Similarity**
- Сравнение топовых слов между запусками
- Диапазон: 0 до 1 (выше лучше)

**Rank-based Stability**
- Сравнение порядка слов в темах
- Метрики: Spearman correlation, Kendall's tau

#### B. Model Robustness

**Bootstrap Stability**
- Метод: Множественные запуски на подвыборках
- Оценка: Консистентность результатов

### 4. 📊 Распределительные Метрики

#### A. Topic Distribution Metrics

**Topic Entropy**
- Формула: H = -Σ p(t) * log(p(t))
- Интерпретация: Выше = более равномерное распределение

**Gini Coefficient**
- Диапазон: 0 до 1
- Интерпретация: 0 = идеальное равенство, 1 = максимальное неравенство

**Topic Coverage**
- Процент документов, отнесенных к темам (vs выбросы)
- Цель: Максимизировать покрытие при сохранении качества

### 5. 🎯 Специализированные Метрики для BERTopic

#### A. BERTopic-специфичные

**Outlier Ratio**
- Процент документов, помеченных как выбросы (-1)
- Интерпретация: Низкий процент обычно лучше

**Topic Embedding Quality**
- Silhouette score для эмбеддингов тем
- Cosine similarity между темами

**Hierarchical Coherence**
- Когерентность на разных уровнях иерархии тем

#### B. HDBSCAN Metrics

**Cluster Validity Index**
- Специфичная для HDBSCAN метрика качества кластеризации

**Stability Score**
- Встроенная в HDBSCAN мера стабильности кластеров

## 🛠️ Практические Рекомендации

### Минимальный Набор Метрик

Для быстрой оценки используйте:
1. **Количество тем** - базовая характеристика
2. **Outlier ratio** - качество покрытия данных
3. **Silhouette score** - качество кластеризации
4. **Coherence score** - семантическое качество тем

### Комплексная Оценка

Для детального анализа добавьте:
1. **Topic stability** - надежность результатов
2. **Human evaluation** - экспертная оценка
3. **Topic diversity** - разнообразие тем
4. **Distribution metrics** - анализ распределения

### Сравнение Моделей

При сравнении разных подходов используйте:
1. **Набор стандартизированных метрик**
2. **Одинаковые данные и предобработку**
3. **Множественные запуски** для оценки стабильности
4. **Экспертную оценку** итоговых результатов

## 📋 Таблица Интерпретации Метрик

| Метрика | Диапазон | Хорошо | Удовлетворительно | Плохо |
|---------|----------|--------|-------------------|-------|
| CV Coherence | 0-1 | >0.5 | 0.3-0.5 | <0.3 |
| Silhouette Score | -1 до +1 | >0.5 | 0.2-0.5 | <0.2 |
| Davies-Bouldin | 0-∞ | <1.0 | 1.0-2.0 | >2.0 |
| Outlier Ratio | 0-1 | <0.1 | 0.1-0.3 | >0.3 |
| Topic Entropy | 0-log(n) | Высокая | Средняя | Низкая |

## 🔧 Инструменты и Библиотеки

### Python Libraries

```python
# Основные библиотеки
from gensim.models.coherencemodel import CoherenceModel  # Coherence metrics
from sklearn.metrics import silhouette_score            # Clustering metrics
from bertopic import BERTopic                          # BERTopic model
from octis.evaluation_metrics import Coherence         # OCTIS evaluation

# Дополнительные инструменты
import pyLDAvis          # Визуализация тем
import wordcloud         # Облака слов
import plotly.express    # Интерактивные визуализации
```

### Специализированные Пакеты

- **OCTIS** - комплексная оценка topic modeling
- **PyLDAvis** - интерактивная визуализация
- **TopicModelingTK** - toolkit для оценки

## 🎬 Примеры Использования

### Базовая Оценка

```python
from src.ml_core.topic_evaluation import quick_evaluate_bertopic

# Быстрая оценка
metrics = quick_evaluate_bertopic(
    topic_model=model,
    texts=texts,
    topics=topics,
    embeddings=embeddings
)

print(f"Количество тем: {metrics['num_topics']}")
print(f"Процент выбросов: {metrics['outlier_percentage']:.1f}%")
print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
print(f"CV Coherence: {metrics['avg_cv_coherence']:.3f}")
```

### Комплексная Оценка

```python
from src.ml_core.topic_evaluation import TopicModelEvaluator

# Создаем оценщик
evaluator = TopicModelEvaluator(model, texts, topics, embeddings)

# Получаем все метрики
results = evaluator.evaluate_comprehensive()

# Генерируем отчет
report = evaluator.generate_evaluation_report("evaluation_report.csv")
```

### Сравнение Моделей

```python
models = [model1, model2, model3]
comparison_results = []

for i, model in enumerate(models):
    topics, _ = model.fit_transform(texts)
    metrics = quick_evaluate_bertopic(model, texts, topics)
    metrics['model'] = f'Model_{i+1}'
    comparison_results.append(metrics)

comparison_df = pd.DataFrame(comparison_results)
print(comparison_df)
```

## 📚 Литература и Источники

1. [Evaluate Topic Model in Python - Towards Data Science](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0)
2. [How to evaluate novel topic modeling method - Medium](https://vtiya.medium.com/how-to-evaluate-novel-topic-modeling-method-104ad9684428)
3. Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures.
4. Newman, D., et al. (2010). Automatic evaluation of topic coherence.

## 🤝 Рекомендации по Использованию

### Этапы Оценки

1. **Предварительная оценка** - базовые метрики для быстрой проверки
2. **Детальный анализ** - комплексная оценка с визуализацией
3. **Экспертная валидация** - привлечение доменных экспертов
4. **Сравнительный анализ** - сопоставление с альтернативными подходами

### Частые Ошибки

❌ **Не делайте:**
- Полагайтесь только на одну метрику
- Игнорируйте качественную оценку
- Оптимизируйте только под автоматические метрики
- Забывайте про стабильность результатов

✅ **Рекомендуется:**
- Используйте комбинацию метрик
- Включайте экспертную оценку
- Проверяйте результаты на разных данных
- Документируйте процесс оценки

---

*Данное руководство регулярно обновляется с учетом новых методов и подходов в области тематического моделирования.*
