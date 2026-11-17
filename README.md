# ДЗ1: ML-модель как веб-сервис (Open Inference Protocol)

Сервис классификации изображений MobileNetV2 (ONNX) с API по спецификации
[Open Inference Protocol](https://github.com/kserve/open-inference-protocol).

## Структура

```
download_model.py    # шаг 1: скачать модель в artifacts/model.onnx
model.py             # обёртка ONNX: загрузка, метаданные, инференс
app.py               # FastAPI-сервис, эндпоинты OIP v2
test_service.py      # smoke-тесты всех эндпоинтов + мини-фаззинг
locustfile.py        # нагрузочное тестирование (доп. балл)
Dockerfile
requirements.txt
.github/workflows/ci.yml   # CI: сборка, тест, публикация в ghcr.io
```

## Локальный запуск

```bash
pip install -r requirements.txt
python download_model.py          # скачать модель
uvicorn app:app --port 8000       # запустить сервис
python test_service.py            # прогнать проверки
```

## Запуск в Docker

```bash
docker build -t mlops-hw1 .
docker run --rm -p 127.0.0.1:8000:8000 --memory=512m --cpus=1 mlops-hw1
python test_service.py
```

## Эндпоинты (OIP v2)

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/v2` | метаданные сервера |
| GET | `/v2/health/live` | liveness |
| GET | `/v2/health/ready` | readiness |
| GET | `/v2/models/mobilenetv2` | метаданные модели |
| GET | `/v2/models/mobilenetv2/ready` | готовность модели |
| POST | `/v2/models/mobilenetv2/infer` | инференс |

Пример запроса:

```bash
curl -X POST http://127.0.0.1:8000/v2/models/mobilenetv2/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[...]}]}'
```

## Образ

`ghcr.io/<логин>/<репо>:latest` — публикуется автоматически GitHub Actions.
