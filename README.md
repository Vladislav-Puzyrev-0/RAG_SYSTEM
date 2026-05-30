Руководство по развёртыванию и эксплуатации системы

---

## 1. Аннотация проекта

Настоящий программный комплекс реализует систему ответов на вопросы по корпусу документов (Retrieval-Augmented Generation, RAG) на основе языковой модели Qwen2.5-7B-Instruct и гибридного поиска FAISS + BM25. Система принимает документы в форматах PDF, DOCX и TXT, выполняет их чанкинг и векторизацию посредством модели MiniLM, после чего обеспечивает семантический поиск и генерацию ответов через REST API, реализованный на FastAPI. Дополнительно поддерживаются потоковая генерация ответов (Server-Sent Events), SQLite-кэширование запросов, генерация образовательных тестов и оффлайн-оценка качества retrieval по метрикам P@k, R@k, MRR, nDCG@k. Комплекс развёртывается на одном сервере без доступа к сети — все веса моделей загружаются из локального кэша HuggingFace.

---

## 2. Требования к окружению

| Компонент | Версия | Назначение |
|---|---|---|
| Python | 3.10 – 3.12 | Среда выполнения |
| PyTorch (torch) | ≥ 2.2 | Инференс LLM и эмбеддингов |
| transformers | 4.57.3 | Загрузка и запуск Qwen2.5-7B-Instruct |
| sentence-transformers | 5.2.0 | Векторизация чанков (MiniLM) |
| faiss-cpu / faiss-gpu | 1.13.2 | HNSW-индекс для dense-поиска |
| FastAPI | 0.135.1 | REST API-сервер |
| uvicorn | 0.38.0 | ASGI-сервер |
| langchain-community | 0.4.1 | Загрузчики документов, FAISS-обёртка |
| pydantic-settings | 2.12.0 | Конфигурация через `.env` |
| bm25s | любая | BM25-индекс (опционально, для гибридного поиска) |
| pymorphy3 | любая | Морфологическая нормализация токенов (опционально) |
| prometheus-fastapi-instrumentator | 7.1.0 | Метрики Prometheus |
| CUDA | ≥ 11.8 | Ускорение на GPU (опционально) |
| ОЗУ | ≥ 16 ГБ | Инференс на CPU; ≥ 8 ГБ VRAM для GPU |
| Дисковое пространство | ≥ 20 ГБ | Веса моделей + индекс |

---

## 3. Установка зависимостей

```bash
# Клонировать репозиторий и перейти в директорию проекта
git clone <url> RAG_SYSTEM
cd RAG_SYSTEM

# Создать и активировать виртуальное окружение
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Установить зависимости
pip install -r requirements.txt

# Создать файл конфигурации (скопировать и отредактировать под своё окружение)
cp .env.example .env

# Создать директорию для документов (если не существует)
mkdir -p documents
```

> **Примечание.** Веса моделей (`Qwen/Qwen2.5-7B-Instruct` и `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) должны быть заранее скачаны в локальный кэш HuggingFace. Переменные окружения `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1` выставляются автоматически при старте (`app/main.py`) и запрещают любые сетевые запросы к HuggingFace Hub.

---

## 4. Пайплайн работы системы

1. **Загрузка документа** → пользователь отправляет файл на `POST /upload`; файл сохраняется в директорию `documents/` → модуль `app/api/upload.py`.

2. **Дедупликация** → вычисляется SHA-256 хеш файла и сравнивается с реестром `vectorstore_index/document_hashes.json`; дубликаты отбрасываются без повторной обработки → модуль `app/rag/ingestion.py`.

3. **Чанкинг** → документ разбивается на перекрывающиеся фрагменты (`CHUNK_SIZE` / `CHUNK_OVERLAP`) с помощью `RecursiveCharacterTextSplitter`; каждому чанку присваиваются метаданные (source, chunk_index) → модуль `app/rag/ingestion.py`.

4. **Векторизация** → чанки батчами кодируются моделью MiniLM (`EmbeddingService`); при `EMBEDDING_NORMALIZE=true` применяется L2-нормализация → модуль `app/rag/embeddings.py`.

5. **Индексация** → векторы добавляются в FAISS HNSW-индекс (inner-product); одновременно строится BM25-индекс по тем же чанкам → модули `app/rag/vectorstore.py`, `app/rag/bm25.py`.

6. **Сохранение** → FAISS-индекс и docstore сохраняются в `vectorstore_index/`; BM25-индекс — в `vectorstore_index/bm25.pkl` → модули `app/rag/vectorstore.py`, `app/rag/bm25.py`.

7. **Приём запроса** → пользователь отправляет вопрос на `POST /ask` или `POST /ask/stream` → модуль `app/api/ask.py`.

8. **Гибридный retrieval** → параллельно выполняются dense-поиск (FAISS) и BM25-поиск; результаты объединяются через Reciprocal Rank Fusion (RRF) → модуль `app/rag/retriever.py`.

9. **Расширение контекста** → к найденным чанкам добавляются соседние чанки того же документа (окно `context_window`) → модуль `app/rag/retrieval.py`.

10. **Генерация ответа** → сформированный контекст и вопрос оборачиваются в ChatML-промпт и передаются Qwen2.5-7B-Instruct; ответ декодируется и возвращается пользователю (или стримится по SSE) → модуль `app/rag/llm.py`.

11. **Кэширование** → готовый ответ записывается в SQLite-таблицу `answers`; повторный идентичный запрос обслуживается из кэша без обращения к LLM → модуль `app/rag/cache.py`.

---

## 5. Запуск

### а) Запуск сервера

```bash
# Запуск через uvicorn напрямую
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Или через точку входа в app/main.py
python app/main.py
```

После старта API доступно по адресу `http://localhost:8000`. Интерактивная документация OpenAPI — `http://localhost:8000/docs`. Метрики Prometheus — `http://localhost:8000/metrics`.

### б) Индексация документов

```bash
# Загрузить один документ
curl -X POST http://localhost:8000/upload \
     -F "files=@/path/to/document.pdf"

# Загрузить несколько документов одновременно
curl -X POST http://localhost:8000/upload \
     -F "files=@doc1.pdf" \
     -F "files=@doc2.docx"

# Фоновая (асинхронная) загрузка — сервер возвращает job_id немедленно
curl -X POST "http://localhost:8000/upload?async_mode=true" \
     -F "files=@large_document.pdf"

# Проверить статус фоновой задачи
curl http://localhost:8000/upload/status/{job_id}
```

### в) Тестовый запрос

```bash
# Синхронный запрос (ожидание полного ответа)
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Что такое гибридный поиск?", "top_k": 3, "mode": "hybrid"}'

# Потоковый запрос (Server-Sent Events)
curl -X POST http://localhost:8000/ask/stream \
     -H "Content-Type: application/json" \
     -d '{"question": "Опишите архитектуру системы", "top_k": 5}'

# Оффлайн-оценка retrieval по golden-датасету
python -m app.eval.run \
    --dataset eval_data/golden.jsonl \
    --mode hybrid \
    --top-k 5 \
    --output results/eval.json

# Управление кэшем
python -m app.rag.cache stats
python -m app.rag.cache clear

# Миграция FAISS-индекса с L2 на inner-product
python -m app.rag.vectorstore migrate-to-ip
```

---

## 6. Описание API-эндпоинтов

| Метод | URL | Параметры | Описание |
|---|---|---|---|
| `GET` | `/health` | — | Проверка работоспособности сервера (не требует инициализации модели) |
| `GET` | `/status` | — | Полный статус системы: устройство, тип индекса, число документов, кэш, BM25 |
| `POST` | `/upload` | `files` (multipart), `async_mode` (bool, default `false`) | Загрузка одного или нескольких документов с дедупликацией |
| `GET` | `/upload/status/{job_id}` | `job_id` (path) | Статус фоновой задачи загрузки |
| `GET` | `/documents` | — | Список загруженных документов с именем, SHA-256 и размером |
| `DELETE` | `/documents/{filename}` | `filename` (path) | Удаление документа из системы (файл + индекс + хеши) |
| `POST` | `/ask` | `question` (str), `top_k` (int), `temperature` (float), `mode` (str), `expand_context` (bool), `context_window` (int), `ef_search` (int, optional) | Синхронный ответ LLM с источниками и диагностикой |
| `POST` | `/ask/stream` | те же, что `/ask` | Потоковый ответ LLM через Server-Sent Events (`token` / `done` / `error`) |
| `POST` | `/quiz/generate` | `topic` (str, optional), `source_filename` (str, optional), `num_questions` (int), `question_types` (list), `difficulty` (str), `language` (str) | Генерация образовательного теста по загруженным материалам |
| `GET` | `/quiz/sources` | — | Список документов, доступных для генерации тестов |
| `GET` | `/metrics` | — | Метрики Prometheus (счётчики запросов, гистограммы латентности, счётчики кэша) |

---

## 7. Структура файлов проекта

```
RAG_SYSTEM/
├── app/
│   ├── api/
│   │   ├── ask.py              # Эндпоинты POST /ask и POST /ask/stream
│   │   ├── documents.py        # Эндпоинты GET /documents и DELETE /documents/{filename}
│   │   ├── health.py           # Эндпоинты GET /health и GET /status
│   │   ├── quiz.py             # Эндпоинты POST /quiz/generate и GET /quiz/sources
│   │   └── upload.py           # Эндпоинт POST /upload и GET /upload/status/{job_id}
│   ├── eval/
│   │   ├── dataset.py          # Загрузка golden-датасета (JSONL)
│   │   ├── latency.py          # Расчёт перцентилей латентности (p50/p95/p99)
│   │   ├── metrics.py          # Метрики retrieval: P@k, R@k, MRR, nDCG@k, Hit@k
│   │   └── run.py              # CLI оффлайн-оценки (python -m app.eval.run)
│   ├── models/
│   │   ├── ask.py              # Pydantic-модели QuestionRequest / AnswerResponse
│   │   ├── common.py           # Общие модели: ErrorResponse, StatusResponse, RetrievedChunk
│   │   ├── documents.py        # Модели DocumentInfo, UploadResponse, DeleteResponse
│   │   └── quiz.py             # Модели QuizRequest, QuizQuestion, QuizResponse
│   ├── rag/
│   │   ├── bm25.py             # BM25Store: индексация, поиск, сериализация
│   │   ├── cache.py            # SQLite-кэш ответов и эмбеддингов (RAGCache)
│   │   ├── embeddings.py       # EmbeddingService поверх SentenceTransformer (MiniLM)
│   │   ├── ingestion.py        # Загрузка, чанкинг, дедупликация документов
│   │   ├── llm.py              # QwenLLM: синхронная и потоковая генерация
│   │   ├── models.py           # Устаревший инициализатор моделей (LangChain pipeline)
│   │   ├── prompts.py          # ChatML-промпты для /ask и /quiz/generate
│   │   ├── quiz.py             # QuizGenerator v3 (новый, поверх HybridRetriever)
│   │   ├── quiz_generator.py   # QuizGenerator v1 (устаревший, поверх LangChain pipeline)
│   │   ├── retrieval.py        # retrieve_and_generate: retrieval → контекст → LLM → кэш
│   │   ├── retriever.py        # HybridRetriever: dense + BM25 + RRF
│   │   ├── system.py           # QwenRAGSystem — главный фасад всех компонентов
│   │   └── vectorstore.py      # FAISS HNSW: создание, загрузка, сохранение, миграция
│   ├── utils/
│   │   └── tokenize_ru.py      # Русский токенизатор для BM25 (regex + pymorphy3)
│   ├── config.py               # Централизованная конфигурация (pydantic-settings)
│   ├── deps.py                 # FastAPI Dependency Injection
│   ├── logging_config.py       # Настройка логирования (console / JSON)
│   ├── main.py                 # Точка входа FastAPI-приложения (lifespan, middleware)
│   ├── metrics.py              # Prometheus-метрики и декораторы инструментации
│   └── middleware.py           # RequestIDMiddleware и глобальный exception handler
├── documents/                  # Директория загружаемых документов (создаётся при старте)
├── vectorstore_index/          # FAISS-индекс, docstore, BM25, SQLite-кэш, хеши (авто)
│   ├── index.faiss
│   ├── index.pkl
│   ├── bm25.pkl
│   ├── cache.sqlite
│   └── document_hashes.json
├── index.html                  # Одностраничный фронтенд (отдаётся на GET /)
├── requirements.txt            # Зафиксированные зависимости
└── .env                        # Конфигурация окружения (не коммитится)
```

---

## 8. Переменные окружения

Все переменные задаются в файле `.env` в корне проекта. Значения по умолчанию соответствуют конфигурации для CPU-сервера без GPU.

| Переменная | Значение по умолчанию | Описание |
|---|---|---|
| `HOST` | `0.0.0.0` | Адрес, на котором слушает uvicorn |
| `PORT` | `8000` | Порт сервера |
| `LOG_LEVEL` | `INFO` | Уровень логирования (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `console` | Формат логов: `console` (текст) или `json` (для ELK-стека) |
| `CORS_ORIGINS` | `["http://localhost:3000","http://localhost:8000"]` | JSON-список разрешённых CORS-источников |
| `DOCS_DIR` | `documents` | Директория для хранения загружаемых файлов |
| `SUPPORTED_EXTENSIONS` | `.pdf,.docx,.doc,.txt` | Допустимые форматы документов |
| `MAX_UPLOAD_MB` | `50` | Максимальный размер загружаемого файла (МБ) |
| `CHUNK_SIZE` | `1000` | Размер текстового чанка (символов) |
| `CHUNK_OVERLAP` | `200` | Перекрытие соседних чанков (символов) |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Имя модели эмбеддингов в HuggingFace Hub |
| `EMBEDDING_DEVICE` | `cpu` | Устройство для эмбеддингов (`cpu` или `cuda`) |
| `EMBEDDING_BATCH_SIZE` | `64` | Размер батча при векторизации документов |
| `EMBEDDING_NORMALIZE` | `true` | Применять L2-нормализацию к векторам |
| `VECTORSTORE_PATH` | `vectorstore_index` | Директория хранения FAISS-индекса и сопутствующих файлов |
| `FAISS_USE_HNSW` | `true` | Использовать HNSW-индекс (иначе — Flat) |
| `FAISS_HNSW_M` | `32` | Параметр M графа HNSW (число рёбер на вершину) |
| `FAISS_HNSW_EF_CONSTRUCTION` | `200` | efConstruction при построении HNSW-индекса |
| `FAISS_HNSW_EF_SEARCH` | `64` | efSearch по умолчанию при поиске |
| `FAISS_USE_GPU` | `true` | Пытаться использовать GPU-сборку FAISS |
| `FAISS_METRIC` | `ip` | Метрика расстояния: `ip` (inner-product) или `l2` |
| `HYBRID_ENABLED` | `true` | Включить гибридный (BM25 + dense) поиск |
| `HYBRID_RRF_K` | `60` | Константа k для Reciprocal Rank Fusion |
| `HYBRID_TOP_K_INTERNAL_FACTOR` | `4` | Множитель top_k для внутреннего поиска каждого канала |
| `BM25_PATH` | `vectorstore_index/bm25.pkl` | Путь к сериализованному BM25-индексу |
| `RAG_CACHE_ENABLED` | `true` | Включить SQLite-кэш ответов и эмбеддингов |
| `RAG_CACHE_PATH` | `vectorstore_index/cache.sqlite` | Путь к файлу SQLite-кэша |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Имя языковой модели в HuggingFace Hub |
| `LLM_MAX_NEW_TOKENS` | `512` | Максимальное число генерируемых токенов |
| `LLM_DEFAULT_TEMPERATURE` | `0.3` | Температура генерации по умолчанию |
| `LLM_DEVICE` | `auto` | Устройство для LLM (`auto`, `cpu`, `cuda`, `cuda:0`) |
| `RETRIEVAL_DEFAULT_TOP_K` | `3` | Число чанков по умолчанию для контекста |
| `RETRIEVAL_MAX_TOP_K` | `20` | Максимально допустимое значение top_k в запросе |
| `DEDUP_ENABLED` | `true` | Включить SHA-256 дедупликацию при загрузке документов |

---

## Ссылка на раздел диплома

Настоящий документ является Приложением Д к выпускной квалификационной работе магистра и соответствует следующим разделам основного текста ВКР:

- **Раздел 2.3** — «Архитектура программного комплекса»: описание модульной структуры системы (пп. 4, 7 настоящего приложения).
- **Раздел 2.5** — «Реализация гибридного retrieval и кэширования»: пайплайн обработки запросов, конфигурация FAISS HNSW и BM25 (пп. 4, 8).
- **Раздел 3.1** — «Развёртывание и эксплуатация экспериментального стенда»: инструкции по установке, запуску и работе с API (пп. 3, 5, 6).

---

*Российский университет дружбы народов (РУДН), 2026.*
