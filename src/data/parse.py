import asyncio
import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение учетных данных из переменных окружения
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")
phone = os.getenv("TELEGRAM_PHONE")
username = os.getenv("TELEGRAM_USERNAME")

# Создание клиента Telegram
client = TelegramClient(
    username,
    api_id,
    api_hash,
    system_version="4.16.30-vxCUSTOM",
    device_model="Desktop",
    app_version="1.0",
)


async def parse_forbes_news(days=21):
    # Подключение к Telegram
    await client.start()
    print("Клиент запущен")

    # Канал Forbes Russia
    channel_username = "forbesrussia"

    # Получение текущей даты и времени
    now = datetime.now()
    # Получение даты и времени 24 часа назад
    day_ago = now - timedelta(days=days)

    # Получение сообщений из канала
    channel_entity = await client.get_entity(channel_username)

    # Список для хранения новостей
    news_data = []

    # Получение истории сообщений
    offset_id = 0
    limit = 100
    total_messages = 0

    while True:
        history = await client(
            GetHistoryRequest(
                peer=channel_entity,
                offset_id=offset_id,
                offset_date=None,
                add_offset=0,
                limit=limit,
                max_id=0,
                min_id=0,
                hash=0,
            )
        )

        if not history.messages:
            break

        messages = history.messages

        for message in messages:
            # Проверка, что сообщение не старше 21 дня
            if message.date.replace(tzinfo=None) < day_ago:
                break

            # Добавление сообщения в список
            news_data.append(
                {
                    "id": message.id,
                    "date": message.date,
                    "text": message.message,
                    "views": getattr(message, "views", 0),
                    "forwards": getattr(message, "forwards", 0),
                }
            )

        # Если мы дошли до сообщений старше 21 дня, выходим из цикла
        if messages[-1].date.replace(tzinfo=None) < day_ago:
            break

        # Обновление смещения для следующего запроса
        offset_id = messages[-1].id
        total_messages += len(messages)

        # Небольшая задержка, чтобы не превысить лимиты API
        await asyncio.sleep(1)

    # Создание DataFrame из собранных данных
    df = pd.DataFrame(news_data)

    # Создание директории для сохранения данных, если она не существует
    os.makedirs("./data/raw", exist_ok=True)

    # Сохранение данных в CSV файл
    df.to_csv("./data/raw/forbes-news.csv", index=False)

    print(f"Собрано {len(df)} новостей за последние {days} дней")
    print("Данные сохранены в ./data/raw/forbes-news.csv")

    # Закрытие клиента
    await client.disconnect()
    return df


# Запуск асинхронной функции
if __name__ == "__main__":
    df = asyncio.run(parse_forbes_news(days=21))
    print(df.head())
