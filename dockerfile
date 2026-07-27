FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and dataset
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run uvicorn server for Python FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]