import sys
import os
import uvicorn

# Pastikan folder root backend terdaftar di Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import FastAPI app dari app/main.py
from app.main import app

# Expose app for ASGI servers (uvicorn, gunicorn, etc.)
__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)

