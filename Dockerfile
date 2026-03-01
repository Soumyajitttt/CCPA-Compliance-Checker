# ---- Base image ----
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Pre-download the model at build time ----
# This avoids downloading during inference (which causes slow first requests)
# google/flan-t5-base is free, public, no HF token needed
RUN python -c "\
from transformers import T5ForConditionalGeneration, T5Tokenizer; \
T5Tokenizer.from_pretrained('google/flan-t5-base'); \
T5ForConditionalGeneration.from_pretrained('google/flan-t5-base'); \
print('Model downloaded successfully!')"

# Copy application code
COPY app/ .

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
