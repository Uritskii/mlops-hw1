"""
Шаг 6 (доп. балл). Нагрузочное тестирование через locust.

Установка (локально, НЕ в образ сервиса):
    pip install locust

Запуск с веб-интерфейсом (открой http://localhost:8089):
    locust -f locustfile.py --host http://127.0.0.1:8000

Или без UI, сразу сценарий (50 пользователей, 2 минуты):
    locust -f locustfile.py --host http://127.0.0.1:8000 \
        --headless -u 50 -r 5 -t 2m --csv report

Для отчёта снимай метрики (RPS, p50/p95/p99, ошибки) при
u = 1, 10, 25, 50, 100 и строй график: где насыщается throughput
и когда начинают расти p99 и 503-е.

Перед тестом запусти сервис с лимитами как у проверяющего:
    docker run --rm -p 8000:8000 --cpus=4 --memory=16g <образ>
"""

import random
from locust import HttpUser, task, constant_throughput

# Готовим вход один раз: 1x3x224x224 float32 — это ~600 КБ JSON,
# не генерируем его на каждый запрос, чтобы мерить сервер, а не клиента.
SHAPE = [1, 3, 224, 224]
SIZE = 1 * 3 * 224 * 224
# Случайные значения честнее константы: исключают кеширующие оптимизации
PAYLOAD = {
    "inputs": [{
        "name": "input",
        "shape": SHAPE,
        "datatype": "FP32",
        "data": [random.random() for _ in range(SIZE)],
    }],
}


class InferenceUser(HttpUser):
    # Каждый виртуальный пользователь держит ~1 RPS (1 запрос в секунду)
    wait_time = constant_throughput(1)

    @task
    def infer(self):
        with self.client.post(
            "/v2/models/mobilenetv2/infer",
            json=PAYLOAD,
            name="/v2/models/mobilenetv2/infer",
            catch_response=True,
        ) as resp:
            # 200 — ок; 503 при перегрузке — ожидаемый управляемый отказ,
            # считаем его отдельно, а не как "failure"
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503:
                resp.success()  # видно в статистике как отдельный случай
            else:
                resp.failure(f"unexpected {resp.status_code}: {resp.text[:200]}")
