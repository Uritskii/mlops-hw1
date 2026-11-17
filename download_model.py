import os
import urllib.request

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.onnx")

SOURCES = [
    # 1. Официальный ONNX Model Zoo (прямая ссылка на validated-модель)
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
    # 2. Зеркало на HuggingFace (onnx-community)
    "https://huggingface.co/onnx-community/mobilenetv2-100/resolve/main/onnx/model.onnx",
]


def download():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"Модель уже скачана: {MODEL_PATH} ({size_mb:.1f} МБ)")
        return

    for url in SOURCES:
        print(f"Скачиваю: {url}")
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
            size_mb = os.path.getsize(MODEL_PATH) / 1e6
            print(f"OK: {MODEL_PATH} ({size_mb:.1f} МБ)")
            return
        except Exception as e:
            print(f"  не получилось: {e}")

    raise RuntimeError("Не удалось скачать модель ни из одного источника")


def check():
    """Проверка: модель читается и сигнатура входов/выходов как ожидается."""
    import onnxruntime as ort

    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("\n--- Сигнатура модели (это пойдёт в метаданные OIP) ---")
    for inp in session.get_inputs():
        print(f"INPUT  name={inp.name!r} shape={inp.shape} type={inp.type}")
    for out in session.get_outputs():
        print(f"OUTPUT name={out.name!r} shape={out.shape} type={out.type}")


if __name__ == "__main__":
    download()
    check()
