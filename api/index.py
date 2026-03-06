"""
Vercel serverless entry point for the FastAPI backend.

Vercel looks for an `app` object in api/index.py.
This re-exports the FastAPI app so it works as a serverless function.
"""

import sys
import os

# Add project root to path so imports work in Vercel's environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi_backend.main import app
