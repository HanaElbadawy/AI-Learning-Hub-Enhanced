"""
01_documents.py
-----------------
Phase 1 — Document Loading (Multi-Source)

بيحوّل كل ملف .txt في أي مجلد فرعي جوه data/raw/ (langchain, huggingface_transformers,
scikit_learn, pytorch, ...) إلى Document object:
    Document(page_content="...", metadata={"source": "...", "title": "...", "site": "..."})

ونحفظهم كلهم في ملف واحد: data/processed/documents.jsonl

تشغيل:
    python 01_documents.py
"""

import json

from langchain_core.documents import Document

from utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from utils.logging_utils import get_logger

logger = get_logger("01_documents")


def load_raw_file(path, site_name: str) -> Document:
    raw_text = path.read_text(encoding="utf-8")

    # أول سطر كتبناه في 00_data_collection.py كان "SOURCE_URL: ..."
    lines = raw_text.split("\n", 2)
    source_url = ""
    body = raw_text
    if lines and lines[0].startswith("SOURCE_URL:"):
        source_url = lines[0].replace("SOURCE_URL:", "").strip()
        body = raw_text.split("\n\n", 1)[1] if "\n\n" in raw_text else ""

    title = path.stem.replace("_", " ")

    return Document(
        page_content=body.strip(),
        metadata={
            "source": source_url or str(path),
            "title": title,
            "file_name": path.name,
            "site": site_name,
        },
    )


def documents_to_jsonl(documents: list[Document], out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in documents:
            record = {"page_content": doc.page_content, "metadata": doc.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    if not DATA_RAW_DIR.exists():
        logger.warning(f"{DATA_RAW_DIR} مش موجود. شغل 00_data_collection.py الأول.")
        return

    site_dirs = sorted(d for d in DATA_RAW_DIR.iterdir() if d.is_dir())
    if not site_dirs:
        logger.warning(
            f"مفيش أي مجلد مصدر جوه {DATA_RAW_DIR}. شغل 00_data_collection.py الأول."
        )
        return

    documents = []
    for site_dir in site_dirs:
        txt_files = sorted(site_dir.glob("*.txt"))
        logger.info(f"بنحمّل {len(txt_files)} ملف من المصدر '{site_dir.name}'")
        for p in txt_files:
            documents.append(load_raw_file(p, site_name=site_dir.name))

    logger.info(f"اتحمّل {len(documents)} document من {len(site_dirs)} مصدر.")

    out_path = DATA_PROCESSED_DIR / "documents.jsonl"
    documents_to_jsonl(documents, out_path)
    logger.info(f"اتحفظوا في {out_path}")


if __name__ == "__main__":
    main()
