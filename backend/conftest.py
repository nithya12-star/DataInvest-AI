"""
conftest.py — ensure the backend/app package is importable during tests.
"""
import sys
import os

# Add backend/ directory to path so `from app.modules.profiler import ...` works
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
