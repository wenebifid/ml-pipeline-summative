from locust import HttpUser, task, between
import os

class APIUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def predict(self):
        img_path = "samples/sample1.jpg"
        if not os.path.exists(img_path):
            return
        with open(img_path, "rb") as f:
            files = {"file": ("sample1.jpg", f, "image/jpeg")}
            self.client.post("/predict", files=files, timeout=30)

    @task(1)
    def health(self):
        self.client.get("/health")
