FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales pour opencv/PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# tensorflow-cpu = beaucoup plus léger que tensorflow (pas de CUDA sur VPS)
COPY api/requirements.txt .
RUN pip install --no-cache-dir tensorflow-cpu==2.18.0 && \
    pip install --no-cache-dir -r requirements.txt

# Code source (api/ + src/ pour data_loader, etc.)
COPY api/ /app/api/
COPY src/ /app/src/

WORKDIR /app/api

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
