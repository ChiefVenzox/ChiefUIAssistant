"""Modeli sıfırdan eğit. Proje kökünden çalıştır:
    python scripts/train_model.py --preset chiefui-30m --data backend/datasets/bin
"""
import _bootstrap  # noqa: F401
from app.training.train import main

if __name__ == "__main__":
    main()
