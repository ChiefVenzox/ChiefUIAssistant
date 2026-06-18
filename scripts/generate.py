"""Terminalden UI üret (test). Proje kökünden çalıştır:
    python scripts/generate.py --instruction "Create a responsive hero section"
"""
import _bootstrap  # noqa: F401
from app.inference.generate import _main

if __name__ == "__main__":
    _main()
