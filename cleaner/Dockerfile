# cleaner/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# (Optional but helpful) system certs + tzdata for HTTPS & timestamps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the cleaner code
COPY cleaner.py minio_io.py duckdb_writer.py cleaning_rules.py ./

# Default command: start the RabbitMQ consumer
CMD ["python", "cleaner.py"]
