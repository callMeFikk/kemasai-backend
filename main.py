"""
Main entry point for uvicorn running from the backend directory:
    uvicorn main:app --reload
"""

from app.main import app, DesignRequest, build_prompt

__all__ = ["app", "DesignRequest", "build_prompt"]
