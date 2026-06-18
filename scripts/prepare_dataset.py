"""JSONL veriyi token bin'lerine çevir. Proje kökünden çalıştır:
    python scripts/prepare_dataset.py --input backend/datasets --out backend/datasets/bin
"""
import _bootstrap  # noqa: F401
from app.training.dataset import main

if __name__ == "__main__":
    main()
