import sys
import os

# Pastikan folder root backend terdaftar di Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import FastAPI app dari app/main.py
# HuggingFace Space akan menjalankan: uvicorn app:app --host 0.0.0.0 --port 7860
from app.main import app  # noqa: F401 — diekspos untuk uvicorn

# Hanya jalankan uvicorn jika dijalankan langsung (lokal dev)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
