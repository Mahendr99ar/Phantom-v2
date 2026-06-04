FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY demo.py .
COPY tests/ ./tests/

# Create data directory for LSM persistence
RUN mkdir -p /app/data

ENV PYTHONPATH=/app/src
ENV PHANTOM_DATA_DIR=/app/data

EXPOSE 8000

CMD ["python", "demo.py"]
