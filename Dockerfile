# --- Базовый образ: slim, чтобы уложиться в лимит 1 ГБ ---
FROM python:3.12-slim

WORKDIR /app

# PYTHONDONTWRITEBYTECODE — не писать .pyc (образ read-only-friendly)
# PYTHONUNBUFFERED — логи сразу в stdout (их видит docker logs)
# Ограничиваем потоки BLAS/OpenMP: иначе каждый воркер гребёт все ядра,
# и потоки конкурируют за CPU (лекция 3)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Слой зависимостей — отдельно и раньше кода:
# пока requirements.txt не менялся, Docker переиспользует кеш слоя.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ТОЛЬКО то, что нужно для работы сервиса
COPY app.py model.py ./
COPY artifacts/ ./artifacts/

# Непривилегированный пользователь (безопасность, лекция 3)
RUN useradd --uid 10001 --create-home app
USER 10001

# EXPOSE документирует порт; публикацию выполняет docker run -p
EXPOSE 8000

# workers=1: каждая копия модели занимает RAM;
# масштабирование — репликами контейнера, а не воркерами внутри.
# timeout-graceful-shutdown — время завершить in-flight запросы при stop.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-graceful-shutdown", "10"]
