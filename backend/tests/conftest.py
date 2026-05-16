"""Shared pytest fixtures and configuration for the backend test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BACKEND_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(BACKEND_ROOT / ".env", override=False)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
