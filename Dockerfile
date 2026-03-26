# Locust LLM load test — single image for master and worker
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY locustfile.py locust.conf ./

# Master/worker pass flags only via compose/K8s command — do not repeat `locust` there
# or argv becomes `locust locust ...` and Locust treats `locust` as a user class name.
ENTRYPOINT ["locust"]
