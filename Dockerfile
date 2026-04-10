FROM python:3.11-slim

WORKDIR /app

# 1. Standardizing environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app 

# 2. Lightweight build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Optimized Caching: Copy requirements from their actual location
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy the entire project
COPY . .

# 5. Ensure Python alias (though usually redundant in this image, it's safe)
RUN ln -sf /usr/local/bin/python3 /usr/local/bin/python

# 6. Expose the Hugging Face default port
EXPOSE 7860

# 7. Start server using the module path
# We use 'python -m uvicorn' to ensure the working directory is in the path
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]