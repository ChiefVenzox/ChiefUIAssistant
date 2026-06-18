"""Kendi BPE tokenizer'ımızı eğit. Proje kökünden çalıştır:
    python scripts/train_tokenizer.py --input backend/datasets --vocab-size 16000
"""
import _bootstrap  # noqa: F401  (sys.path ayarı)
from app.tokenizer.train_tokenizer import main

if __name__ == "__main__":
    main()
