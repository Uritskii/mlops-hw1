"""
Обёртка над ONNX-моделью: загрузка, метаданные, инференс.

Отделён от app.py, чтобы HTTP-слой не знал про onnxruntime,
а модельный слой — про FastAPI.
"""

import numpy as np
import onnxruntime as ort

# Маппинг типов numpy <-> типов Open Inference Protocol.
# Полный набор OIP: BOOL, UINT8..64, INT8..64, FP16/32/64, BYTES.
NP_TO_OIP = {
    np.dtype("float32"): "FP32",
    np.dtype("float64"): "FP64",
    np.dtype("float16"): "FP16",
    np.dtype("int8"): "INT8",
    np.dtype("int16"): "INT16",
    np.dtype("int32"): "INT32",
    np.dtype("int64"): "INT64",
    np.dtype("uint8"): "UINT8",
    np.dtype("uint16"): "UINT16",
    np.dtype("uint32"): "UINT32",
    np.dtype("uint64"): "UINT64",
    np.dtype("bool"): "BOOL",
}
OIP_TO_NP = {v: k for k, v in NP_TO_OIP.items()}

# Маппинг строковых типов ONNX (например "tensor(float)") -> numpy
ONNX_TYPE_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}


class ModelBundle:
    """Загруженная ONNX-сессия + метаданные входов/выходов."""

    def __init__(self, model_path: str):
        # CPUExecutionProvider — инференс на CPU (GPU в задании нет).
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        # Метаданные берём из самого графа — так сигнатура всегда честная,
        # и фаззер проверяющего сервера получит корректное описание.
        self.inputs_meta = [self._tensor_meta(t) for t in self.session.get_inputs()]
        self.outputs_meta = [self._tensor_meta(t) for t in self.session.get_outputs()]

    @staticmethod
    def _tensor_meta(t) -> dict:
        """Преобразует описание тензора onnxruntime в метаданные OIP."""
        np_dtype = np.dtype(ONNX_TYPE_TO_NP[t.type])
        # В OIP переменные размерности обозначаются -1.
        # В ONNX они могут быть строками ("batch") или None.
        shape = [-1 if (isinstance(d, str) or d is None) else int(d) for d in t.shape]
        return {
            "name": t.name,
            "datatype": NP_TO_OIP[np_dtype],
            "shape": shape,
            "_np_dtype": np_dtype,  # служебное поле, наружу не отдаём
        }

    def warmup(self):
        """Прогрев: один контрольный инференс нулевым батчем.

        Делает две вещи: (1) убеждаемся, что модель реально считает,
        (2) убираем «холодный» первый запрос из-за JIT/аллокаций.
        """
        inputs = {}
        for meta in self.inputs_meta:
            shape = [1 if d == -1 else d for d in meta["shape"]]
            inputs[meta["name"]] = np.zeros(shape, dtype=meta["_np_dtype"])
        self.infer(inputs)

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Запуск инференса. inputs: {имя_входа: numpy-массив}."""
        out_names = [o.name for o in self.session.get_outputs()]
        result = self.session.run(out_names, inputs)
        return dict(zip(out_names, result))
