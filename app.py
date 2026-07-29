import sys
import os

# Pastikan folder root terdaftar di Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Expose FastAPI app — HuggingFace Spaces akan menjalankan file ini sebagai app
from app.main import app
