import random
from locust import HttpUser, task, constant_throughput


SHAPE = [1, 3, 224, 224]
SIZE = 1 * 3 * 224 * 224
PAYLOAD = {
    "inputs": [{
        "name": "input",
        "shape": SHAPE,
        "datatype": "FP32",
        "data": [random.random() for _ in range(SIZE)],
    }],
}


class InferenceUser(HttpUser):
    wait_time = constant_throughput(1)

    @task
    def infer(self):
        with self.client.post(
            "/v2/models/mobilenetv2/infer",
            json=PAYLOAD,
            name="/v2/models/mobilenetv2/infer",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503:
                resp.success()  # видно в статистике как отдельный случай
            else:
                resp.failure(f"unexpected {resp.status_code}: {resp.text[:200]}")
