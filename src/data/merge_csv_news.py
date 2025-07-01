#!/usr/bin/env python3
"""
Скрипт для объединения всех CSV файлов из каталога data/raw/ в один файл.
Обрабатывает различные форматы дат и удаляет дубликаты.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Настройка логирования


def normalize_date_format(date_str, logger):
    """
    Нормализует различные форматы дат к единому формату YYYY-MM-DD HH:MM:SS
    """
    if pd.isna(date_str) or date_str == "":
        return None

    try:
        # Пробуем разные форматы дат
        formats = [
            "%Y-%m-%d %H:%M:%S.%f",  # 2025-04-19 11:47:57.515222
            "%Y-%m-%d %H:%M:%S",  # 2025-04-18 6:32:09
            "%Y-%m-%d",  # 2025-05-09
            "%d.%m.%Y",  # 09.05.2025
            "%d.%m.%Y %H:%M:%S",  # 09.05.2025 12:30:45
        ]

        for fmt in formats:
            try:
                parsed_date = datetime.strptime(str(date_str), fmt)
                return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

        # Если не удалось распарсить, возвращаем исходное значение
        logger.warning(f"Не удалось распарсить дату: {date_str}")
        return str(date_str)

    except Exception as e:
        logger.error(f"Ошибка при обработке даты {date_str}: {e}")
        return str(date_str)


def standardize_columns(df, filename, logger):
    """
    Приводит колонки DataFrame к стандартному формату
    """
    logger.info(f"Обрабатываю файл: {filename}")
    logger.info(f"Исходные колонки: {list(df.columns)}")

    # Создаем словарь маппинга колонок для разных форматов
    column_mapping = {}

    # Если есть колонки как в csv-dump.csv
    if "global_post_id" in df.columns:
        column_mapping = {
            "title": "title",
            "text": "text",
            "date": "date",
            "link": "url",
        }
        # Добавляем дополнительные поля для csv-dump
        df["source"] = filename

    # Если есть колонки как в lenta-news файлах
    elif "url" in df.columns:
        column_mapping = {
            "title": "title",
            "text": "text",
            "date": "date",
            "url": "url",
        }
        df["source"] = filename

    # Стандартизируем названия колонок
    df_standardized = pd.DataFrame()

    for std_col, orig_col in column_mapping.items():
        if orig_col in df.columns:
            df_standardized[std_col] = df[orig_col]
        else:
            df_standardized[std_col] = None

    # Добавляем источник если его нет
    if "source" not in df_standardized.columns:
        df_standardized["source"] = filename
    else:
        df_standardized["source"] = filename

    return df_standardized


def merge_csv_files(raw_data_path, output_path, logger):
    """
    Основная функция для объединения CSV файлов
    """

    logger.info("Начинаю объединение CSV файлов...")

    # Путь к каталогу с исходными файлами
    raw_data_path = Path(raw_data_path)

    # Путь для сохранения результата
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    # Список для хранения всех DataFrame
    all_dataframes = []

    # Получаем все CSV файлы (исключаем .gitkeep)
    csv_files = [f for f in raw_data_path.glob("*.csv") if f.name != ".gitkeep"]

    logger.info(f"Найдено CSV файлов: {len(csv_files)}")

    for csv_file in csv_files:
        try:
            logger.info(f"Обрабатываю файл: {csv_file.name}")

            # Читаем CSV файл
            df = pd.read_csv(csv_file, encoding="utf-8")
            logger.info(f"Загружено строк: {len(df)}")

            if len(df) == 0:
                logger.warning(f"Файл {csv_file.name} пустой, пропускаю")
                continue

            # Стандартизируем колонки
            df_std = standardize_columns(df, csv_file.name, logger)

            # Нормализуем даты
            if "date" in df_std.columns:
                logger.info("Нормализую формат дат...")
                df_std["date"] = df_std["date"].apply(
                    normalize_date_format, logger=logger
                )

            all_dataframes.append(df_std)
            logger.info(f"Файл {csv_file.name} обработан успешно")

        except Exception as e:
            logger.error(f"Ошибка при обработке файла {csv_file.name}: {e}")
            continue

    if not all_dataframes:
        logger.error("Не удалось загрузить ни одного файла!")
        return

    # Объединяем все DataFrame
    logger.info("Объединяю все данные...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"Общее количество строк после объединения: {len(combined_df)}")

    # Удаляем дубликаты
    logger.info("Удаляю дубликаты...")

    # Определяем колонки для поиска дубликатов
    duplicate_cols = ["title", "text", "date"]
    # Используем только те колонки, которые есть в данных
    duplicate_cols = [col for col in duplicate_cols if col in combined_df.columns]

    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=duplicate_cols, keep="first")
    final_count = len(combined_df)

    logger.info(f"Удалено дубликатов: {initial_count - final_count}")
    logger.info(f"Итоговое количество уникальных записей: {final_count}")

    # Сортируем по дате
    if "date" in combined_df.columns:
        logger.info("Сортирую по дате...")
        combined_df = combined_df.sort_values("date")

    # Сохраняем результат
    output_file = output_path / "merged_news_data.csv"
    logger.info(f"Сохраняю результат в файл: {output_file}")

    combined_df.to_csv(output_file, index=False, encoding="utf-8")

    # Выводим статистику
    logger.info("=== СТАТИСТИКА ОБЪЕДИНЕНИЯ ===")
    logger.info(f"Обработано файлов: {len(csv_files)}")
    logger.info(f"Итоговое количество записей: {len(combined_df)}")
    logger.info(f"Колонки в итоговом файле: {list(combined_df.columns)}")

    if "date" in combined_df.columns:
        date_range = combined_df["date"].dropna()
        if len(date_range) > 0:
            logger.info(f"Диапазон дат: {date_range.min()} - {date_range.max()}")

    if "source" in combined_df.columns:
        source_stats = combined_df["source"].value_counts()
        logger.info("Распределение по источникам:")
        for source, count in source_stats.items():
            logger.info(f"  {source}: {count} записей")

    logger.info(f"Файл сохранен: {output_file}")

    return output_file


if __name__ == "__main__":
    raw_data_path = Path(r"..\data\raw").as_posix()
    output_path = Path(r"..\data\processed").as_posix()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        result_file = merge_csv_files(raw_data_path, output_path, logger)
        print("\n✅ Объединение завершено успешно!")
        print(f"📁 Результат сохранен в: {result_file}")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"\n❌ Ошибка при выполнении: {e}")
