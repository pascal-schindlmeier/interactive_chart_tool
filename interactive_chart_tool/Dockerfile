FROM python:3.11-slim

# Optional: kleine Tools, u.a. für Healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends wget tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt installieren
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Quellcode kopieren
COPY app/ ./app/

EXPOSE 8000

# Healthcheck ist optional, funktioniert nur wenn dein Backend /healthz anbietet
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD wget -q -O- http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.ict_backend:app", "--host", "0.0.0.0", "--port", "8000"]
