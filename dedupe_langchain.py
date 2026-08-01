"""
dedupe_langchain.py
----------------------
أداة تُستخدم مرة واحدة بس: بتدمج data/raw/langchain_rag و
data/raw/langchain_agents جوه data/raw/langchain (لأن الـ 3 مصادر كان
عندهم نفس allowed_path_prefix، فاللي اتجمع غالبًا نفس الصفحات مكرر).

بتحتفظ بنسخة واحدة بس من كل ملف فريد، وبتسألك قبل ما تمسح المجلدات
الزيادة (مفيش حذف تلقائي من غير تأكيد).

تشغيل مرة واحدة بس:
    python dedupe_langchain.py

بعد ما تشتغل بنجاح، احذفي الملف ده نفسه (مش جزء من الـ pipeline الدائم).
"""

import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
PRIMARY = RAW_DIR / "langchain"
REDUNDANT = [RAW_DIR / "langchain_rag", RAW_DIR / "langchain_agents"]


def main():
    if not PRIMARY.exists():
        print(f"❌ مفيش {PRIMARY} - وقفت. شغّلي 00_data_collection.py الأول.")
        return

    primary_files = {p.name for p in PRIMARY.glob("*.txt")}
    print(f"📂 'langchain' فيها {len(primary_files)} ملف حاليًا.\n")

    total_merged = 0
    total_duplicate = 0

    for folder in REDUNDANT:
        if not folder.exists():
            print(f"⏭️  {folder.name} مش موجودة، تجاهلتها.")
            continue

        files = list(folder.glob("*.txt"))
        merged_here = 0
        dup_here = 0

        for f in files:
            if f.name in primary_files:
                dup_here += 1  # موجود في langchain بالفعل، مش هننسخه تاني
            else:
                shutil.copy2(f, PRIMARY / f.name)
                primary_files.add(f.name)
                merged_here += 1

        print(
            f"📁 {folder.name}: {len(files)} ملف إجمالي | "
            f"✅ {merged_here} جديد اتنقل | ⏭️ {dup_here} مكرر اتجاهل"
        )
        total_merged += merged_here
        total_duplicate += dup_here

    print(f"\n{'=' * 50}")
    print(f"النتيجة: {total_merged} ملف فريد اتضاف، {total_duplicate} ملف مكرر اتجاهل.")
    print(f"إجمالي الملفات في 'langchain' دلوقتي: {len(list(PRIMARY.glob('*.txt')))}")
    print(f"{'=' * 50}\n")

    confirm = input("امسح مجلدات langchain_rag و langchain_agents دلوقتي؟ (yes/no): ")
    if confirm.strip().lower() == "yes":
        for folder in REDUNDANT:
            if folder.exists():
                shutil.rmtree(folder)
                print(f"🗑️  اتمسح: {folder}")
        print("\n✅ خلصنا. دلوقتي كمّلي من 01_documents.py عادي.")
    else:
        print("\nمسحتش حاجة. شغّلي السكريبت تاني وقولي yes لما تكوني جاهزة.")


if __name__ == "__main__":
    main()
