import re

from razdel import tokenize

import nltk

try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords

stop_words = stopwords.words("russian")


def tokenize_ru(text):
    words = tokenize(text)
    return [word.text for word in words]


def clean_text(text):
    # Приведение текста к нижнему регистру
    # text = text.lower()s
    # text = emoji.replace_emoji(text, '')
    # Замена всех не-словесных символов на пробел (кроме букв и знаков препинания)
    text = re.sub(r"\W+", " ", text)

    # Удаление URL-адресов
    text = re.sub(r"http\S+", "", text)

    # Создание шаблона для HTML-тегов
    html = re.compile(r"&lt;.*?&gt;")

    # Удаление HTML-тегов из текста
    text = html.sub(r"", text)

    # Список пунктуаций для удаления
    punctuations = "@#!?+&amp;*[]-%.:/();$=&gt;&lt;|{}^" + "'`" + "_"
    for p in punctuations:
        text = text.replace(p, "")  # Удаление пунктуации

    # Удаление стоп-слов и приведение слов к нижнему регистру
    text = [word for word in text.split() if word.lower() not in stop_words]

    # Объединение слов обратно в текст
    text = " ".join(text)

    # Создание шаблона для поиска эмодзи
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # эмоции
        "\U0001f300-\U0001f5ff"  # символы и пиктограммы
        "\U0001f680-\U0001f6ff"  # транспорт и карты
        "\U0001f1e0-\U0001f1ff"  # флаги
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )

    # Удаление эмодзи из текста
    text = emoji_pattern.sub(r"", text)

    return text
