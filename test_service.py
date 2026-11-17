"""
Шаг 2 (проверка). Smoke-тесты всех эндпоинтов OIP + имитация фаззинга.

Перед запуском подними сервис (одним из способов):
    uvicorn app:app --port 8000
или в контейнере:
    docker run --rm -p 127.0.0.1:8000:8000 mlops-hw1

Запуск тестов:
    python test_service.py

Если всё зелёное — сервис готов к сдаче.
"""

import sys
import requests

BASE = "http://127.0.0.1:8000"
MODEL = "mobilenetv2"

passed = 0
failed = 0


def check(title: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {title}")
    else:
        failed += 1
        print(f"  FAIL  {title}  {detail}")


def main():
    # --- 1. Метаданные сервера ---
    r = requests.get(f"{BASE}/v2")
    check("GET /v2 -> 200", r.status_code == 200, r.text)
    body = r.json()
    check("server metadata: есть name/version/extensions",
          all(k in body for k in ("name", "version", "extensions")), str(body))

    # --- 2. Health ---
    r = requests.get(f"{BASE}/v2/health/live")
    check("GET /v2/health/live -> live=true", r.json().get("live") is True, r.text)
    r = requests.get(f"{BASE}/v2/health/ready")
    check("GET /v2/health/ready -> ready=true", r.json().get("ready") is True, r.text)

    # --- 3. Метаданные модели (по ним фаззер строит запросы!) ---
    r = requests.get(f"{BASE}/v2/models/{MODEL}")
    check("GET /v2/models/{model} -> 200", r.status_code == 200, r.text)
    meta = r.json()
    check("model metadata: есть inputs/outputs",
          "inputs" in meta and "outputs" in meta, str(meta))
    inp = meta["inputs"][0]
    print(f"    вход: {inp['name']} {inp['datatype']} {inp['shape']}")
    check("model ready", requests.get(f"{BASE}/v2/models/{MODEL}/ready")
          .json().get("ready") is True)

    # --- 4. Корректный infer ---
    # Формируем вход по сигнатуре из метаданных: -1 заменяем на батч=1
    shape = [1 if d == -1 else d for d in inp["shape"]]
    size = 1
    for d in shape:
        size *= d
    payload = {
        "id": "smoke-1",
        "inputs": [{
            "name": inp["name"],
            "shape": shape,
            "datatype": inp["datatype"],
            "data": [0.5] * size,  # плоский массив — легальный формат OIP
        }],
    }
    r = requests.post(f"{BASE}/v2/models/{MODEL}/infer", json=payload)
    check("POST infer (корректный) -> 200", r.status_code == 200, r.text[:300])
    if r.status_code == 200:
        out = r.json()
        check("infer: вернулся id из запроса", out.get("id") == "smoke-1")
        check("infer: есть outputs", len(out.get("outputs", [])) > 0)
        o = out["outputs"][0]
        print(f"    выход: {o['name']} {o['datatype']} {o['shape']}")

    # --- 5. Фаззинг: кривые запросы должны давать 4xx, а не падение ---
    bad_cases = {
        "не JSON": ("raw", "{not json"),
        "нет inputs": ("json", {}),
        "пустой inputs": ("json", {"inputs": []}),
        "неизвестный вход": ("json", {"inputs": [{"name": "nope", "shape": [1],
                                                  "datatype": "FP32", "data": [1.0]}]}),
        "неверный datatype": ("json", {"inputs": [{"name": inp["name"], "shape": shape,
                                                   "datatype": "FP128", "data": [0.0] * size}]}),
        "неверная shape": ("json", {"inputs": [{"name": inp["name"], "shape": [2, 2],
                                                "datatype": inp["datatype"], "data": [1, 2, 3, 4]}]}),
        "data короче shape": ("json", {"inputs": [{"name": inp["name"], "shape": shape,
                                                   "datatype": inp["datatype"], "data": [0.5]}]}),
    }
    for title, (kind, body) in bad_cases.items():
        if kind == "raw":
            r = requests.post(f"{BASE}/v2/models/{MODEL}/infer", data=body,
                              headers={"Content-Type": "application/json"})
        else:
            r = requests.post(f"{BASE}/v2/models/{MODEL}/infer", json=body)
        ok = r.status_code in (400, 404, 422) and "error" in r.text
        check(f"фаззинг [{title}] -> 4xx + error", ok, f"got {r.status_code}")

    # --- 6. Несуществующая модель -> 404 ---
    r = requests.post(f"{BASE}/v2/models/unknown_model/infer", json=payload)
    check("неизвестная модель -> 404", r.status_code == 404, f"got {r.status_code}")

    print(f"\nИтого: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
