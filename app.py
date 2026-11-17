"""

Эндпоинты (спека: github.com/kserve/open-inference-protocol):
  GET  /v2                                         — метаданные сервера
  GET  /v2/health/live                             — liveness (процесс жив)
  GET  /v2/health/ready                            — readiness (все модели готовы)
  GET  /v2/models/{name}[/versions/{ver}]          — метаданные модели
  GET  /v2/models/{name}[/versions/{ver}]/ready    — готовность модели
  POST /v2/models/{name}[/versions/{ver}]/infer    — инференс

"""

import os
import time
import logging
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from model import ModelBundle, NP_TO_OIP, OIP_TO_NP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oip-server")

# --- Константы сервиса 
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.onnx")
MODEL_NAME = os.environ.get("MODEL_NAME", "mobilenetv2")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
SERVER_NAME = "oip-fastapi-server"
SERVER_VERSION = "1.0.0"

# Семафор: не более N одновременных инференсов на процесс.
# Защита от перегрузки: лишние запросы получают управляемый 503,
MAX_PARALLEL_INFER = int(os.environ.get("MAX_PARALLEL_INFER", "4"))
infer_slots = threading.BoundedSemaphore(MAX_PARALLEL_INFER)


# ---------------------------------------------------------------------------
# Жизненный цикл: загрузка -> прогрев -> приём запросов
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Загрузка модели из %s ...", MODEL_PATH)
    bundle = ModelBundle(MODEL_PATH)
    t0 = time.time()
    bundle.warmup()
    log.info("Модель загружена и прогрета за %.2f с", time.time() - t0)

    app.state.bundle = bundle
    app.state.ready = True 
    yield
    app.state.ready = False
    app.state.bundle = None


app = FastAPI(title=SERVER_NAME, version=SERVER_VERSION, lifespan=lifespan)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def error_response(status: int, message: str) -> JSONResponse:
    """Ошибка по спеке OIP: тело {"error": "<описание>"}."""
    return JSONResponse(status_code=status, content={"error": message})


def check_model(app: FastAPI, model_name: str):
    """Проверка, что запрошенная модель существует (иначе 404 по OIP)."""
    if model_name != MODEL_NAME:
        return error_response(404, f"model '{model_name}' not found")
    if not getattr(app.state, "ready", False):
        return error_response(503, "model is not ready")
    return None


def parse_input_tensor(raw: dict, meta: dict) -> np.ndarray:
    """Валидация и преобразование одного входного тензора из JSON в numpy.

    Бросает ValueError с понятным сообщением — превратим его в HTTP 400.
    Проверяем: наличие полей, datatype, shape (с учётом -1), длину data.
    """
    name = raw.get("name")
    if not name:
        raise ValueError("input tensor: missing 'name'")
    if "data" not in raw:
        raise ValueError(f"input '{name}': missing 'data'")
    if "shape" not in raw or "datatype" not in raw:
        raise ValueError(f"input '{name}': missing 'shape' or 'datatype'")

    # 1) Тип данных
    req_dtype = raw["datatype"]
    if req_dtype not in OIP_TO_NP:
        raise ValueError(f"input '{name}': unsupported datatype '{req_dtype}'")
    np_dtype = OIP_TO_NP[req_dtype]

    # 2) Форма: OIP разрешает -1 для переменных размерностей.
    #    Сверяем ранг и статические размерности с ожидаемыми из модели.
    req_shape = list(raw["shape"])
    exp_shape = meta["shape"]
    if len(req_shape) != len(exp_shape):
        raise ValueError(
            f"input '{name}': rank mismatch, got {req_shape}, expected {exp_shape}"
        )
    for got, exp in zip(req_shape, exp_shape):
        if not isinstance(got, int) or got < 0:
            raise ValueError(f"input '{name}': invalid shape dim {got}")
        if exp != -1 and got != exp:
            raise ValueError(
                f"input '{name}': shape mismatch {req_shape} vs expected {exp_shape}"
            )

    # 3) Данные: спека разрешает плоский или вложенный массив.
    #    np.asarray + reshape приводит оба варианта к req_shape.
    try:
        arr = np.asarray(raw["data"], dtype=np_dtype)
    except (ValueError, TypeError) as e:
        raise ValueError(f"input '{name}': cannot convert data to {req_dtype}: {e}")

    expected_size = int(np.prod(req_shape)) if req_shape else 1
    if arr.size != expected_size:
        raise ValueError(
            f"input '{name}': data has {arr.size} elements, shape {req_shape} "
            f"requires {expected_size}"
        )
    return arr.reshape(req_shape)


# ---------------------------------------------------------------------------
# Эндпоинты OIP
# ---------------------------------------------------------------------------
@app.get("/v2")
def server_metadata():
    return {"name": SERVER_NAME, "version": SERVER_VERSION, "extensions": []}


@app.get("/v2/health/live")
def server_live():

    return {"live": True}


@app.get("/v2/health/ready")
def server_ready():
   
    if getattr(app.state, "ready", False):
        return {"ready": True}
    return error_response(503, "server is not ready")


def model_metadata_payload() -> dict:
    bundle = app.state.bundle
    return {
        "name": MODEL_NAME,
        "versions": [MODEL_VERSION],
        "platform": "onnx_onnxv1",  
        "inputs": [
            {k: m[k] for k in ("name", "datatype", "shape")}
            for m in bundle.inputs_meta
        ],
        "outputs": [
            {k: m[k] for k in ("name", "datatype", "shape")}
            for m in bundle.outputs_meta
        ],
    }


@app.get("/v2/models/{model_name}")
@app.get("/v2/models/{model_name}/versions/{model_version}")
def model_metadata(model_name: str, model_version: str | None = None):
    if err := check_model(app, model_name):
        return err
    if model_version is not None and model_version != MODEL_VERSION:
        return error_response(404, f"model version '{model_version}' not found")
    return model_metadata_payload()


@app.get("/v2/models/{model_name}/ready")
@app.get("/v2/models/{model_name}/versions/{model_version}/ready")
def model_ready(model_name: str, model_version: str | None = None):
    if err := check_model(app, model_name):
        return err
    return {"name": MODEL_NAME, "ready": True}


@app.post("/v2/models/{model_name}/infer")
@app.post("/v2/models/{model_name}/versions/{model_version}/infer")
async def infer(model_name: str, request: Request, model_version: str | None = None):
    if err := check_model(app, model_name):
        return err
    if model_version is not None and model_version != MODEL_VERSION:
        return error_response(404, f"model version '{model_version}' not found")


    try:
        body = await request.json()
    except Exception:
        return error_response(400, "request body is not valid JSON")

    raw_inputs = body.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        return error_response(400, "request must contain non-empty 'inputs' array")

    bundle: ModelBundle = app.state.bundle
    expected = {m["name"]: m for m in bundle.inputs_meta}

    input_arrays: dict[str, np.ndarray] = {}
    for raw in raw_inputs:
        try:
            name = raw.get("name")
            if name not in expected:
                raise ValueError(f"unknown input '{name}', expected: {list(expected)}")
            input_arrays[name] = parse_input_tensor(raw, expected[name])
        except ValueError as e:
            # Ошибка клиента -> 400, процесс жив, фаззинг продолжается
            return error_response(400, str(e))

    # --- Ограничение параллелизма: перегрузка -> управляемый 503 ---
    if not infer_slots.acquire(blocking=False):
        return error_response(503, "inference capacity exhausted")

    try:
        t0 = time.time()
        outputs = bundle.infer(input_arrays)
        duration_ms = (time.time() - t0) * 1000
    except Exception as e:
      
        log.exception("Inference failed")
        return error_response(500, f"inference failed: {e}")
    finally:
        infer_slots.release()


    requested = body.get("outputs")
    if requested:
        wanted = {o.get("name") for o in requested}
        unknown = wanted - set(outputs)
        if unknown:
            return error_response(400, f"unknown outputs requested: {sorted(unknown)}")
        outputs = {k: v for k, v in outputs.items() if k in wanted}

  
    response_outputs = []
    for name, arr in outputs.items():
        arr = np.asarray(arr)
        response_outputs.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "datatype": NP_TO_OIP.get(arr.dtype, "FP32"),
                "data": arr.flatten().tolist(),
            }
        )

    log.info("infer ok model=%s duration_ms=%.1f", MODEL_NAME, duration_ms)

    response = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "outputs": response_outputs,
    }
    if "id" in body:
        response["id"] = body["id"]
    return response
