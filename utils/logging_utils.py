"""
utils/logging_utils.py
-----------------------
- get_logger(): logger عادي لكل مرحلة (يطبع بتوقيت وواضح مين بعت الرسالة).
- log_failure(): بيسجل صف واحد في evaluation/failure_log.csv كل ما مرحلة
  تفشل (retrieval / context / prompt / generation)، بنفس الفكرة اللي
  اتعلمناها في Lab 9 (فصل نوع الفشل حسب الطبقة).
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from utils.config import EVALUATION_DIR

FAILURE_LOG_PATH = EVALUATION_DIR / "failure_log.csv"
_FAILURE_FIELDS = ["timestamp", "query", "failed_layer", "expected", "got", "notes"]


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_failure(
    query: str,
    failed_layer: str,
    expected: str = "",
    got: str = "",
    notes: str = "",
):
    """
    failed_layer لازم يكون واحد من:
    'retrieval' | 'context' | 'prompt' | 'generation'
    """
    valid_layers = {"retrieval", "context", "prompt", "generation"}
    if failed_layer not in valid_layers:
        raise ValueError(f"failed_layer لازم يكون واحد من {valid_layers}")

    is_new_file = not FAILURE_LOG_PATH.exists()
    with open(FAILURE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FAILURE_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "failed_layer": failed_layer,
                "expected": expected,
                "got": got,
                "notes": notes,
            }
        )
