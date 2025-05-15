import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm
import time
import threading
import sys
import json

# Добавляем путь к корневому каталогу проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Импортируем функции для работы с таблицей новостных файлов
from src.database.database import get_news_file, save_news_file, update_news_file_status

# Загрузка переменных окружения из .env файла
load_dotenv()

# Глобальный замок для безопасного вывода в консоль из разных потоков
print_lock = threading.Lock()

# Создаем словарь для хранения результатов парсинга по датам
parsed_results = {}
results_lock = threading.Lock()


def safe_print(message):
    """Потокобезопасный вывод в консоль"""
    with print_lock:
        print(message)


def parse_date(current_date, headers, progress=None):
    """
    Парсинг новостей с сайта Lenta.ru за конкретную дату.
    Эта функция будет выполняться в отдельном потоке для каждой даты.

    Args:
        current_date (datetime): Дата для парсинга
        headers (dict): Заголовки для HTTP-запроса
        progress (tqdm): Объект прогресс-бара для обновления

    Returns:
        list: Список новостей за указанную дату
        str: Путь к сохраненному CSV-файлу
    """
    date_results = []
    
    try:
        # Проверяем, есть ли уже файл с новостями за эту дату
        news_file = get_news_file("lenta", current_date)
        
        # Если файл существует и его статус "ready", просто возвращаем путь к файлу
        if news_file and news_file["status"] == "ready":
            safe_print(f"Найден существующий файл с новостями за {current_date.strftime('%Y-%m-%d')}: {news_file['file_path']}")
            
            # Обновляем прогресс-бар
            if progress:
                progress.update(1)
            
            # Возвращаем пустой список (данные будут загружены из файла) и путь к файлу
            return [], news_file["file_path"]
        
        # Если файл находится в процессе парсинга, возвращаем ошибку
        if news_file and news_file["status"] == "parsing":
            safe_print(f"Файл с новостями за {current_date.strftime('%Y-%m-%d')} в процессе парсинга другим процессом")
            
            # Обновляем прогресс-бар
            if progress:
                progress.update(1)
            
            return [], None
        
        # Если файл существует, но его статус "error", или файла нет, начинаем парсинг
        if news_file:
            # Обновляем статус на "parsing"
            update_news_file_status(news_file["id"], "parsing")
            file_id = news_file["id"]
        else:
            # Создаем новую запись в базе данных
            output_filename = f"./data/raw/lenta-news_{current_date.strftime('%Y%m%d')}.csv"
            save_news_file(
                source="lenta",
                news_date=current_date,
                file_path=output_filename,
                news_count=0,
                status="parsing"
            )
            # Получаем созданную запись
            news_file = get_news_file("lenta", current_date)
            file_id = news_file["id"]
        
        date_str_for_url = current_date.strftime('%Y/%m/%d')
        url_for_date = f"https://lenta.ru/{date_str_for_url}/" # Новый, подтвержденный формат URL

        safe_print(f"Парсинг новостей за {current_date.strftime('%Y-%m-%d')} с URL: {url_for_date}")

        try:
            response = requests.get(url_for_date, headers=headers, timeout=10)
            if response.status_code == 404:
                safe_print(f"Страница не найдена (404) для даты {current_date.strftime('%Y-%m-%d')} по URL: {url_for_date}")
                # Попробуем альтернативные URL-форматы для раздела новостей и статей
                alternative_urls = [
                    f"https://lenta.ru/news/{date_str_for_url}/",  # Раздел новостей
                    f"https://lenta.ru/articles/{date_str_for_url}/"  # Раздел статей
                ]
                found_alternative = False
                
                for alt_url in alternative_urls:
                    try:
                        alt_response = requests.get(alt_url, headers=headers, timeout=10)
                        if alt_response.status_code == 200:
                            safe_print(f"Найден альтернативный URL: {alt_url}")
                            url_for_date = alt_url
                            response = alt_response
                            found_alternative = True
                            break
                    except Exception as e:
                        safe_print(f"Ошибка при проверке альтернативного URL {alt_url}: {e}")
                
                if not found_alternative:
                    # Обновляем статус на "error"
                    update_news_file_status(file_id, "error", {"error": "Страница не найдена для всех проверенных URL"})
                    
                    # Обновляем прогресс-бар
                    if progress:
                        progress.update(1)
                        
                    return [], None  # Переход к следующей дате
            
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Обновленные селекторы для поиска новостей на архивных страницах
            # Используем разные селекторы и объединяем результаты
            news_blocks = []
            
            # 1. Селекторы для основной архивной страницы
            archive_selectors = [
                'a.card-fullnews._news', 
                'a.card-mini._topnews', 
                'li.parts-page__item > a',
                'a.card', 
                'li.archive-page__item > a',    # Типичный селектор для архивных страниц
                'div.archive-page__item a',     # Более общий селектор
                'div.item a[href^="/news/"]',   # Ссылки на новости
                'div.item a[href^="/articles/"]',  # Ссылки на статьи
                'div.rubric-page__item a'       # Элементы на странице рубрики
            ]
            
            for selector in archive_selectors:
                blocks = soup.select(selector)
                if blocks:
                    news_blocks.extend(blocks)
                    safe_print(f"Найдено {len(blocks)} новостных блоков с селектором '{selector}'")
            
            # 2. Поиск по ссылкам на новости и статьи (часто используется в новых версиях сайтов)
            news_links = soup.select('a[href*="/news/"], a[href*="/articles/"]')
            if news_links:
                # Фильтруем, чтобы не включать навигационные ссылки
                filtered_links = [
                    link for link in news_links 
                    if link.get('href') and 
                    ('/' + date_str_for_url.replace('/', '/') in link.get('href') or  # Проверка на текущую дату в URL
                     len(link.text.strip()) > 20)  # Заголовки обычно длиннее, чем навигационные элементы
                ]
                safe_print(f"Найдено {len(filtered_links)} ссылок на новости/статьи с нужной датой или длинным текстом")
                news_blocks.extend(filtered_links)
            
            # 3. Если до сих пор ничего не нашли, попробуем найти все статьи/новости без привязки к дате
            if not news_blocks:
                safe_print(f"Селекторы не нашли новостные блоки. Пробуем общий поиск статей и новостей...")
                general_links = soup.select('a')
                news_blocks = [
                    link for link in general_links 
                    if link.get('href') and 
                    ('/news/' in link.get('href') or '/articles/' in link.get('href')) and
                    len(link.text.strip()) > 20  # Исключаем короткие навигационные ссылки
                ]
                safe_print(f"Найдено {len(news_blocks)} потенциальных новостных ссылок через общий поиск")

            if not news_blocks:
                safe_print(f"Новостные блоки не найдены на {url_for_date} с использованием всех доступных селекторов.")
                
                # Последняя попытка - проверить, есть ли структура разделов новостей и статей напрямую
                news_section_urls = [
                    f"https://lenta.ru/news/{current_date.strftime('%Y/%m/%d')}/",
                    f"https://lenta.ru/articles/{current_date.strftime('%Y/%m/%d')}/"
                ]
                
                for section_url in news_section_urls:
                    try:
                        section_response = requests.get(section_url, headers=headers, timeout=10)
                        if section_response.status_code == 200:
                            safe_print(f"Найден раздел новостей: {section_url}")
                            section_soup = BeautifulSoup(section_response.text, "html.parser")
                            
                            # Ищем ссылки в разделе
                            section_links = section_soup.select('a')
                            potential_news = [
                                link for link in section_links 
                                if link.get('href') and 
                                ('/news/' in link.get('href') or '/articles/' in link.get('href')) and
                                len(link.text.strip()) > 20
                            ]
                            
                            if potential_news:
                                safe_print(f"Найдено {len(potential_news)} ссылок в разделе {section_url}")
                                news_blocks.extend(potential_news)
                    except Exception as e:
                        safe_print(f"Ошибка при проверке раздела {section_url}: {e}")

            if not news_blocks:
                safe_print(f"После всех попыток не найдено новостных блоков для даты {current_date.strftime('%Y-%m-%d')}.")
                # Обновляем статус на "ready", но с метаданными о том, что новостей нет
                update_news_file_status(file_id, "ready", {"note": "Новости не найдены", "news_count": 0})
                
                # Создаем пустой CSV для этой даты
                output_filename = f"./data/raw/lenta-news_{current_date.strftime('%Y%m%d')}.csv"
                # Создаем пустой DataFrame с нужными колонками
                empty_df = pd.DataFrame(columns=["title", "text", "date", "url"])
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                empty_df.to_csv(output_filename, index=False, encoding='utf-8')
                
                # Обновляем информацию о файле
                save_news_file(
                    source="lenta",
                    news_date=current_date,
                    file_path=output_filename,
                    news_count=0,
                    status="ready",
                    metadata={"note": "Новости не найдены"}
                )
                
                # Обновляем прогресс-бар
                if progress:
                    progress.update(1)
                
                return [], output_filename
            
            for block in news_blocks:
                try:
                    title = ""
                    # Попытка извлечь заголовок из разных возможных структур
                    # 1. Сначала проверяем, не является ли сам текст ссылки заголовком
                    link_text = block.text.strip()
                    if len(link_text) > 20:  # Предполагаем, что заголовок достаточно длинный
                        title = link_text
                    else:
                        # 2. Ищем заголовок внутри блока
                        title_element_full = block.select_one('h3.card-fullnews__title, span.card-fullnews__title, h2, h3') 
                        if title_element_full:
                            title = title_element_full.text.strip()
                        else:
                            title_element_mini = block.select_one('h3.card-mini__title, span.card-mini__title') 
                            if title_element_mini:
                                title = title_element_mini.text.strip()

                    # Если заголовок не найден, пропускаем этот блок
                    if not title:
                        continue

                    # Получаем URL новости
                    news_url_relative = block.get("href")
                    news_url = None
                    if news_url_relative:
                        if news_url_relative.startswith("http"):
                            news_url = news_url_relative
                        elif news_url_relative.startswith("/"):
                            news_url = "https://lenta.ru" + news_url_relative
                        else:
                            continue
                    else: 
                        continue

                    # Проверяем, что URL действительно указывает на новость или статью
                    if "/news/" not in news_url and "/articles/" not in news_url:
                        continue

                    news_text, news_date_obj = "", current_date 

                    if news_url:
                        news_content = get_full_news_content(news_url, headers)
                        if news_content and news_content[0]: 
                            news_text, news_date_obj_from_article = news_content
                            if isinstance(news_date_obj_from_article, datetime):
                                news_date_obj = news_date_obj_from_article
                        else:
                            news_text = "Не удалось загрузить текст статьи"

                    # Проверка, что новость относится к текущему дню
                    if news_date_obj.date() != current_date.date():
                        continue  # Пропускаем новости с неправильной датой

                    date_results.append(
                        {
                            "title": title,
                            "text": news_text,
                            "date": news_date_obj,
                            "url": news_url,
                        }
                    )

                except Exception as e:
                    safe_print(f"Ошибка при обработке новостного блока (URL: {news_url if 'news_url' in locals() and news_url else 'не определен'}): {e}")
            
            safe_print(f"Завершили обработку даты {current_date.strftime('%Y-%m-%d')}. Собрано {len(date_results)} новостей.")

            # Сохраняем новости в CSV файл
            if date_results:
                df = pd.DataFrame(date_results)
                # Убедимся, что все ожидаемые колонки присутствуют
                expected_columns = ["title", "text", "date", "url"]
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = pd.NA
                
                # Преобразуем даты
                if 'date' in df.columns and not df['date'].empty:
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                
                # Формируем путь к файлу
                output_filename = f"./data/raw/lenta-news_{current_date.strftime('%Y%m%d')}.csv"
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                
                # Сохраняем файл
                df.to_csv(output_filename, index=False, encoding='utf-8')
                
                # Обновляем запись в базе данных
                save_news_file(
                    source="lenta",
                    news_date=current_date,
                    file_path=output_filename,
                    news_count=len(df),
                    status="ready"
                )
                
                safe_print(f"Сохранены новости за {current_date.strftime('%Y-%m-%d')} в файл {output_filename}")
            else:
                # Создаем пустой CSV с нужной структурой
                output_filename = f"./data/raw/lenta-news_{current_date.strftime('%Y%m%d')}.csv"
                empty_df = pd.DataFrame(columns=["title", "text", "date", "url"])
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                empty_df.to_csv(output_filename, index=False, encoding='utf-8')
                
                # Обновляем запись в базе данных
                save_news_file(
                    source="lenta",
                    news_date=current_date,
                    file_path=output_filename,
                    news_count=0,
                    status="ready",
                    metadata={"note": "Новости не найдены"}
                )
                
                safe_print(f"Создан пустой файл для даты {current_date.strftime('%Y-%m-%d')}: {output_filename}")

        except requests.exceptions.RequestException as e:
            safe_print(f"Ошибка HTTP-запроса для даты {current_date.strftime('%Y-%m-%d')} ({url_for_date}): {e}")
            # Обновляем статус на "error"
            update_news_file_status(file_id, "error", {"error": str(e)})
            
            # Возвращаем пустой список и None вместо пути к файлу
            if progress:
                progress.update(1)
            return [], None
            
        except Exception as e:
            safe_print(f"Произошла ошибка при парсинге Lenta.ru для даты {current_date.strftime('%Y-%m-%d')}: {e}")
            # Обновляем статус на "error"
            update_news_file_status(file_id, "error", {"error": str(e)})
            
            # Возвращаем пустой список и None вместо пути к файлу
            if progress:
                progress.update(1)
            return [], None
        
        # Обновляем прогресс-бар
        if progress:
            progress.update(1)
        
        return date_results, output_filename
        
    except Exception as e:
        safe_print(f"Критическая ошибка при обработке даты {current_date.strftime('%Y-%m-%d')}: {e}")
        if progress:
            progress.update(1)
        return [], None


def parse_lenta_news(start_date: datetime, end_date: datetime, max_workers=10, progress_callback=None):
    """
    Парсинг новостей с сайта Lenta.ru за указанный период дат с использованием многопоточности.
    Каждая дата обрабатывается в отдельном потоке и сохраняется в отдельный CSV файл.

    Args:
        start_date (datetime): Начальная дата для парсинга.
        end_date (datetime): Конечная дата для парсинга.
        max_workers (int): Максимальное количество потоков.
        progress_callback (callable, optional): Функция обратного вызова для отслеживания прогресса. 
            Принимает один аргумент - значение прогресса от 0 до 1.

    Returns:
        pd.DataFrame: DataFrame с новостями за весь период
    """
    # Заголовки для имитации браузера
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Генерируем список всех дат в указанном диапазоне
    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date)
        current_date += timedelta(days=1)
    
    all_news_data = []
    successful_file_paths = []
    
    # Показываем индикатор прогресса
    print(f"Начинаем многопоточный парсинг Lenta.ru за период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
    print(f"Количество дат для обработки: {len(date_range)}")
    print(f"Используем до {max_workers} параллельных потоков")
    print("Начинаем парсинг, это может занять некоторое время...")
    
    # Счетчик для отслеживания прогресса
    processed_dates = 0
    total_dates = len(date_range)
    
    # Функция для обновления прогресса
    def update_progress():
        nonlocal processed_dates
        processed_dates += 1
        if progress_callback:
            progress = processed_dates / total_dates
            progress_callback(progress)
    
    # Создаем прогресс-бар с общим количеством дат для обработки
    with tqdm(total=len(date_range), desc="Парсинг дат", unit="дата") as progress:
        # Запускаем многопоточную обработку
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создаем список задач - по одной задаче на каждую дату
            futures = {
                executor.submit(
                    parse_date, date, headers, progress
                ): date for date in date_range
            }
            
            # Обрабатываем результаты по мере их готовности
            for future in concurrent.futures.as_completed(futures):
                date = futures[future]
                try:
                    date_results, file_path = future.result()
                    if file_path:
                        successful_file_paths.append(file_path)
                    if date_results:
                        all_news_data.extend(date_results)
                    
                    # Обновляем счетчик прогресса
                    update_progress()
                    
                except Exception as e:
                    safe_print(f"Ошибка при получении результатов для даты {date.strftime('%Y-%m-%d')}: {e}")
                    # Даже при ошибке мы считаем дату обработанной для прогресс-бара
                    update_progress()

    # Если у нас есть успешно обработанные файлы, загружаем данные из них
    if successful_file_paths:
        safe_print(f"Загружаем данные из {len(successful_file_paths)} успешно обработанных файлов")
        
        # Создаем пустой DataFrame для объединения данных
        df_combined = pd.DataFrame()
        
        # Загружаем данные из каждого файла и объединяем их
        for i, file_path in enumerate(successful_file_paths):
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    # Преобразуем даты
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    df_combined = pd.concat([df_combined, df], ignore_index=True)
                
                # Обновляем прогресс загрузки
                if progress_callback:
                    load_progress = (i + 1) / len(successful_file_paths)
                    progress_callback(0.8 + (load_progress * 0.2))  # От 80% до 100%
                    
            except Exception as e:
                safe_print(f"Ошибка при загрузке данных из файла {file_path}: {e}")
        
        # Фильтруем по датам, чтобы точно получить только нужный диапазон
        if not df_combined.empty and 'date' in df_combined.columns:
            df_combined = df_combined[
                (df_combined['date'].dt.date >= start_date.date()) & 
                (df_combined['date'].dt.date <= end_date.date())
            ]
        
        # Удаляем дубликаты
        if not df_combined.empty and 'url' in df_combined.columns:
            df_combined = df_combined.drop_duplicates(subset=['url'], keep='first').reset_index(drop=True)
        
        safe_print(f"Всего загружено {len(df_combined)} новостей за период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        # Сообщаем о завершении (100%)
        if progress_callback:
            progress_callback(1.0)
            
        return df_combined
    
    # Если нет успешно обработанных файлов, но есть данные в памяти
    if all_news_data:
        df = pd.DataFrame(all_news_data)
        expected_columns = ["title", "text", "date", "url"]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = pd.NA
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        safe_print(f"Всего собрано {len(df)} новостей за период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        # Сообщаем о завершении (100%)
        if progress_callback:
            progress_callback(1.0)
            
        return df
    
    # Если нет ни файлов, ни данных в памяти
    safe_print(f"Не найдено новостей с Lenta.ru за период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
    
    # Сообщаем о завершении (100%), даже если новости не найдены
    if progress_callback:
        progress_callback(1.0)
        
    return pd.DataFrame(columns=["title", "text", "date", "url"])


def get_full_news_content(url, headers):
    """
    Получение полного текста новости и даты публикации.

    Args:
        url (str): URL новости
        headers (dict): Заголовки для HTTP-запроса

    Returns:
        tuple: (текст новости, дата публикации)
    """
    try:
        response = requests.get(url, headers=headers, timeout=15)  # Увеличенный таймаут для статей
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Поиск текста новости
        # Более общий подход: сначала находим основное содержимое статьи (content, main, article...)
        news_text = ""
        article_container_selectors = [
            'article',
            'div.topic-body',
            'div.js-topic__text',
            'div.b-topic__content',
            'article[itemprop="articleBody"]',
            'div.article-body',
            'div.content',
            'div.js-material-text',
            'div.material-text'
        ]
        
        article_body = None
        for selector in article_container_selectors:
            article_body = soup.select_one(selector)
            if article_body:
                break
        
        if article_body:
            # Собираем текст из всех <p> внутри найденного блока, ИЛИ из других подходящих тегов, если <p> нет
            paragraphs = article_body.find_all('p', recursive=True)  # Изменено на recursive=True для поиска на всех уровнях
            if not paragraphs: 
                 paragraphs = article_body.find_all(['p', 'div'], class_=lambda x: x != 'topic-body__title' if x else True)
            
            if paragraphs:
                news_text = " ".join([p.text.strip() for p in paragraphs if p.text])
            
            if not news_text.strip():
                 news_text = article_body.text.strip()
        else: 
            # Если основной блок текста не найден, пробуем отдельные селекторы
            text_blocks_selectors = [
                "p.topic-body__content-text",
                "p.article__text",
                "div.article__content p", 
                "div.text-article p"
            ]
            
            for text_selector in text_blocks_selectors:
                text_blocks = soup.select(text_selector)
                if text_blocks:
                    news_text = " ".join([block.text.strip() for block in text_blocks])
                    break
            
            if not news_text:
                # Последняя попытка - просто взять весь текст из body и удалить скрипты, стили и т.д.
                body = soup.find('body')
                if body:
                    # Удаляем скрипты, стили и комментарии
                    for element in body(['script', 'style', 'header', 'footer', 'nav']):
                        element.extract()
                    
                    # Берем весь оставшийся текст
                    news_text = body.text.strip()
                    news_text = ' '.join(news_text.split())  # Нормализуем пробелы

        # Поиск даты публикации
        date_selector = (
            'time.topic-header__time[datetime], ' 
            'time.g-date[datetime], '
            'time[itemprop="datePublished"][datetime], '
            'div.topic-header__item.topic-header__time[datetime], '
            'span.topic-header__time[datetime],'
            'time[datetime]'  # Любой time с datetime
        )
        date_element = soup.select_one(date_selector)
        
        if not date_element:
            # Расширенный поиск даты публикации
            date_selectors = [
                "time.topic-header__time",
                "time.g-date", 
                "time[itemprop='datePublished']",
                ".article__date",
                ".news-article__date",
                ".publication-date",
                "meta[name='pubdate']",
                "meta[name='date']",
                "meta[property='article:published_time']"
            ]
            
            for date_sel in date_selectors:
                date_element = soup.select_one(date_sel)
                if date_element:
                    break

        news_date_obj = datetime.now() 

        if date_element:
            datetime_attr = date_element.get("datetime") or date_element.get("content")
            date_text_content = date_element.text.strip()
            
            if datetime_attr:
                try:
                    if 'Z' in datetime_attr:
                        datetime_attr = datetime_attr.replace("Z", "+00:00")
                    if len(datetime_attr) == 19 and 'T' in datetime_attr: # Простой формат YYYY-MM-DDTHH:MM:SS
                         datetime_attr += "+00:00" # Добавляем смещение для fromisoformat
                    news_date_obj = datetime.fromisoformat(datetime_attr)
                except ValueError:
                    pass
            elif date_text_content:
                # Пытаемся распознать дату из текста (например, "14 мая 2023, 15:30")
                try:
                    # Список возможных месяцев на русском
                    months_ru = {
                        'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6,
                        'июля': 7, 'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12,
                        'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'май': 5, 'июн': 6,
                        'июл': 7, 'авг': 8, 'сен': 9, 'окт': 10, 'ноя': 11, 'дек': 12
                    }
                    
                    # Примеры форматов: "14 мая 2023", "14 мая 2023, 15:30", "14.05.2023"
                    import re
                    
                    # Попытка найти дату в формате "DD месяц YYYY"
                    pattern1 = r'(\d{1,2})\s+([а-яА-Я]+)\s+(\d{4})'
                    match1 = re.search(pattern1, date_text_content)
                    if match1:
                        day, month_str, year = match1.groups()
                        month = months_ru.get(month_str.lower())
                        if month:
                            # Проверяем, есть ли время в строке
                            time_pattern = r'(\d{1,2}):(\d{2})'
                            time_match = re.search(time_pattern, date_text_content)
                            if time_match:
                                hour, minute = time_match.groups()
                                news_date_obj = datetime(int(year), month, int(day), int(hour), int(minute))
                            else:
                                news_date_obj = datetime(int(year), month, int(day))
                    
                    # Попытка найти дату в формате "DD.MM.YYYY"
                    if news_date_obj == datetime.now():  # Если первый паттерн не сработал
                        pattern2 = r'(\d{1,2})[./](\d{1,2})[./](\d{4})'
                        match2 = re.search(pattern2, date_text_content)
                        if match2:
                            day, month, year = match2.groups()
                            # Проверяем, есть ли время в строке
                            time_pattern = r'(\d{1,2}):(\d{2})'
                            time_match = re.search(time_pattern, date_text_content)
                            if time_match:
                                hour, minute = time_match.groups()
                                news_date_obj = datetime(int(year), int(month), int(day), int(hour), int(minute))
                            else:
                                news_date_obj = datetime(int(year), int(month), int(day))
                except Exception:
                    pass
        else:
            # Если не нашли явное указание даты, попытаемся извлечь её из URL
            try:
                import re
                url_date_pattern = r'/(\d{4})/(\d{2})/(\d{2})/'
                url_date_match = re.search(url_date_pattern, url)
                if url_date_match:
                    year, month, day = url_date_match.groups()
                    news_date_obj = datetime(int(year), int(month), int(day))
            except Exception:
                pass

        return news_text.strip(), news_date_obj

    except Exception as e:
        return "", datetime.now()


if __name__ == "__main__":
    # Пример использования: парсинг новостей за последние 3 дня, включая сегодняшний
    end_date_main = datetime.now()
    start_date_main = end_date_main - timedelta(days=2) 
    
    print(f"Запуск многопоточного парсинга Lenta.ru с {start_date_main.strftime('%Y-%m-%d')} по {end_date_main.strftime('%Y-%m-%d')}")
    
    # Указываем количество потоков (можно настроить под свою систему)
    max_threads = 8  # Разумное количество потоков для большинства систем
    
    df_news = parse_lenta_news(start_date_main, end_date_main, max_workers=max_threads)
    
    if not df_news.empty:
        print(df_news.head())
        print(f"Всего новостей: {len(df_news)}")
        print("Распределение по датам:")
        print(df_news['date'].dt.date.value_counts().sort_index())
    else:
        print("Новостей не найдено.")
