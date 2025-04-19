import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()


def parse_lenta_news(days=1):
    """
    Парсинг новостей с сайта Lenta.ru за указанное количество дней.

    Args:
        days (int): Количество дней для парсинга (по умолчанию 1 - текущий день)

    Returns:
        pd.DataFrame: DataFrame с новостями
    """
    # URL главной страницы Lenta.ru
    url = "https://lenta.ru/"

    # Заголовки для имитации браузера
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Отправка GET-запроса
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Проверка на ошибки HTTP

        # Парсинг HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Список для хранения новостей
        news_data = []

        # Поиск всех новостных блоков
        news_blocks = soup.find_all("a", class_="card-mini _topnews")

        for block in news_blocks:
            try:
                # Получение заголовка
                title_element = block.find("h3", class_="card-mini__title")
                title = title_element.text.strip() if title_element else "Нет заголовка"

                # Получение ссылки на полную новость
                news_url = (
                    url.rstrip("/") + block["href"] if block.has_attr("href") else None
                )

                # Получение текста и даты из полной новости
                news_text, news_date = "", datetime.now()

                if news_url:
                    news_content = get_full_news_content(news_url, headers)
                    if news_content:
                        news_text, news_date = news_content

                # Добавление новости в список
                news_data.append(
                    {
                        "title": title,
                        "text": news_text,
                        "date": news_date,
                        "url": news_url,
                    }
                )

            except Exception as e:
                print(f"Ошибка при обработке новостного блока: {e}")

        # Создание DataFrame из собранных данных
        df = pd.DataFrame(news_data)

        # Создание директории для сохранения данных, если она не существует
        os.makedirs("./data/raw", exist_ok=True)

        # Сохранение данных в CSV файл
        df.to_csv("./data/raw/lenta-news.csv", index=False)

        print(f"Собрано {len(df)} новостей с Lenta.ru")
        print("Данные сохранены в ./data/raw/lenta-news.csv")

        return df

    except Exception as e:
        print(f"Произошла ошибка при парсинге Lenta.ru: {e}")
        return pd.DataFrame()


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
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Поиск текста новости
        text_blocks = soup.find_all("p", class_="topic-body__content-text")
        news_text = " ".join([block.text.strip() for block in text_blocks])

        # Поиск даты публикации
        date_element = soup.find("time", class_="topic-header__time")
        if date_element and date_element.has_attr("datetime"):
            news_date = datetime.fromisoformat(
                date_element["datetime"].replace("Z", "+00:00")
            )
        else:
            news_date = datetime.now()

        return news_text, news_date

    except Exception as e:
        print(f"Ошибка при получении полной новости {url}: {e}")
        return "", datetime.now()


if __name__ == "__main__":
    df = parse_lenta_news()
    print(df.head())
