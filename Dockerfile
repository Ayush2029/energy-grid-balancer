FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better caching)
COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy rest of code
COPY . .

# Ensure python alias exists
RUN ln -sf /usr/local/bin/python3 /usr/local/bin/python

EXPOSE 7860

# Start server (HF-safe)
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]