FROM python:3.9-slim

LABEL maintainer="RVC ONNX Exporter"
LABEL description="HTTP API for batch converting RVC .pth model files to ONNX format"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/exported_onnx /app/uploaded_pth

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "api.py"]
