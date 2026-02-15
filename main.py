"""CLI-приложение для работы с Pinecone и OpenAI embeddings."""
import argparse
import json
import logging
import sys
from pathlib import Path

import openai
from pinecone import Pinecone

from config import Config


# Настройка логирования
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def setup_clients():
    """Инициализирует клиенты OpenAI и Pinecone."""
    try:
        Config.validate()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    try:
        openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        pinecone_client = Pinecone(api_key=Config.PINECONE_API_KEY)
        return openai_client, pinecone_client
    except Exception as e:
        logger.error(f"Ошибка инициализации клиентов: {e}")
        sys.exit(1)


def get_embeddings(texts, client, model):
    """Получает embeddings для списка текстов через OpenAI."""
    try:
        logger.info(f"Получение embeddings для {len(texts)} текстов...")
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        logger.info(f"Успешно получено {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Ошибка при получении embeddings: {e}")
        raise


def ingest_data():
    """Загружает данные из phrases.json в Pinecone."""
    logger.info("Запуск режима загрузки данных (ingest)")
    
    # Чтение данных
    phrases_path = Path("data/phrases.json")
    if not phrases_path.exists():
        logger.error(f"Файл {phrases_path} не найден")
        sys.exit(1)
    
    try:
        with open(phrases_path, "r", encoding="utf-8") as f:
            phrases = json.load(f)
        logger.info(f"Загружено {len(phrases)} записей из {phrases_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка чтения файла: {e}")
        sys.exit(1)
    
    # Инициализация клиентов
    openai_client, pinecone_client = setup_clients()
    
    # Подключение к индексу
    try:
        index = pinecone_client.Index(Config.PINECONE_INDEX_NAME)
        logger.info(f"Подключено к индексу: {Config.PINECONE_INDEX_NAME}")
    except Exception as e:
        logger.error(f"Ошибка подключения к индексу Pinecone: {e}")
        sys.exit(1)
    
    # Подготовка данных для загрузки
    texts = [item["text"] for item in phrases]
    
    # Получение embeddings
    try:
        embeddings = get_embeddings(texts, openai_client, Config.OPENAI_EMBED_MODEL)
    except Exception:
        sys.exit(1)
    
    # Формирование векторов для Pinecone
    vectors = []
    for phrase, embedding in zip(phrases, embeddings):
        vector = {
            "id": phrase["id"],
            "values": embedding,
            "metadata": {
                "text": phrase["text"],
                "topic": phrase.get("topic", "unknown")
            }
        }
        vectors.append(vector)
    
    # Загрузка в Pinecone
    try:
        logger.info(f"Загрузка {len(vectors)} векторов в Pinecone...")
        # Pinecone поддерживает batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            logger.info(f"Загружено {min(i + batch_size, len(vectors))}/{len(vectors)} векторов")
        
        logger.info("Данные успешно загружены в Pinecone")
        print(f"Успешно загружено {len(vectors)} записей в Pinecone")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных в Pinecone: {e}")
        sys.exit(1)


def semantic_search(query, openai_client, pinecone_index, top_k=5):
    """Выполняет semantic search в Pinecone."""
    try:
        # Получение embedding для запроса
        logger.info(f"Получение embedding для запроса...")
        response = openai_client.embeddings.create(
            model=Config.OPENAI_EMBED_MODEL,
            input=[query]
        )
        query_embedding = response.data[0].embedding
        
        # Поиск в Pinecone
        logger.info(f"Поиск в Pinecone (top_k={top_k})...")
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches
    except Exception as e:
        logger.error(f"Ошибка при semantic search: {e}")
        raise


def ask_mode(query, openai_client, pinecone_index):
    """Режим ask: поиск + ответ GPT-4."""
    try:
        # 1. Semantic search
        matches = semantic_search(query, openai_client, pinecone_index, top_k=5)
        
        if not matches:
            print("--- Найдено ---")
            print("Результаты не найдены")
            return
        
        # 2. Вывод найденных результатов
        print("--- Найдено ---")
        for match in matches:
            print(f"[{match.id}] {match.score:.4f}")
            text = match.metadata.get("text", "")
            print(text)
            print()
        
        # 3. Подготовка контекста для GPT-4
        sources_text = "Источники:\n"
        for match in matches:
            text = match.metadata.get("text", "")
            # Обрезаем до 500 символов
            truncated_text = text[:500] + ("..." if len(text) > 500 else "")
            sources_text += f"[{match.id}] {truncated_text}\n"
        
        # 4. Запрос к GPT-4
        logger.info("Отправка запроса в GPT-4...")
        response = openai_client.chat.completions.create(
            model=Config.OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Отвечай кратко, только по источникам. Указывай id источников."
                },
                {
                    "role": "user",
                    "content": f"{query}\n\n{sources_text}"
                }
            ]
        )
        
        answer = response.choices[0].message.content
        
        # 5. Вывод ответа
        print("--- Ответ модели ---")
        print(answer)
        print()
        
    except Exception as e:
        logger.error(f"Ошибка в режиме ask: {e}")
        print(f"Ошибка: {e}")


def interactive_mode():
    """Запускает интерактивный режим (REPL)."""
    # Инициализация клиентов
    openai_client, pinecone_client = setup_clients()
    
    try:
        index = pinecone_client.Index(Config.PINECONE_INDEX_NAME)
        logger.info(f"Подключено к индексу: {Config.PINECONE_INDEX_NAME}")
    except Exception as e:
        logger.error(f"Ошибка подключения к индексу Pinecone: {e}")
        sys.exit(1)
    
    print("Введите запрос (или exit): ", end="", flush=True)
    
    try:
        while True:
            query = input().strip()
            
            if query.lower() in ("exit", "quit", "q"):
                print("Выход из программы")
                break
            
            if not query:
                print("Введите запрос (или exit): ", end="", flush=True)
                continue
            
            # Выполнение поиска и ответа
            ask_mode(query, openai_client, index)
            print("Введите запрос (или exit): ", end="", flush=True)
    
    except KeyboardInterrupt:
        print("\nВыход из программы")
    except EOFError:
        print("\nВыход из программы")


def run_queries():
    """Выполняет поиск для всех запросов из queries.json."""
    logger.info("Запуск режима пакетной обработки запросов (run-queries)")
    
    # Чтение запросов
    queries_path = Path("data/queries.json")
    if not queries_path.exists():
        logger.error(f"Файл {queries_path} не найден")
        sys.exit(1)
    
    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        logger.info(f"Загружено {len(queries)} запросов из {queries_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка чтения файла: {e}")
        sys.exit(1)
    
    # Инициализация клиентов
    openai_client, pinecone_client = setup_clients()
    
    try:
        index = pinecone_client.Index(Config.PINECONE_INDEX_NAME)
        logger.info(f"Подключено к индексу: {Config.PINECONE_INDEX_NAME}")
    except Exception as e:
        logger.error(f"Ошибка подключения к индексу Pinecone: {e}")
        sys.exit(1)
    
    # Обработка каждого запроса
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Запрос {i}/{len(queries)}: {query}")
        print(f"{'='*60}\n")
        
        try:
            ask_mode(query, openai_client, index)
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            print(f"Ошибка при обработке запроса: {e}\n")
    
    print(f"\n{'='*60}")
    print(f"Обработано {len(queries)} запросов")
    print(f"{'='*60}")


def main():
    """Главная функция приложения."""
    parser = argparse.ArgumentParser(
        description="CLI-приложение для работы с Pinecone и OpenAI embeddings"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["ingest", "run-queries"],
        help="Команда для выполнения"
    )
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_data()
    elif args.command == "run-queries":
        run_queries()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()

