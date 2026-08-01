# 📚 AI Learning Hub

AI-powered RAG Assistant لأسئلة الـ AI / NLP / RAG، بيرد من **مواقع documentation
متعددة** (LangChain, Sentence-Transformers, FAISS, scikit-learn, NLTK) مع
ذكر المصدر، وفيه طبقة تقييم وتتبع أخطاء منفصلة.

## 🏗️ الـ Pipeline

```
Documentation Sites (sources.json)
      │
      ▼
00_data_collection.py      → data/raw/<source_name>/*.txt  (لكل مصدر مجلده)
      │
      ▼
01_documents.py             → data/processed/documents.jsonl
      │
      ▼
02_preprocessing.py         → data/processed/documents_clean.jsonl
      │
      ▼
03_chunking.py               → data/processed/chunks.jsonl
      │
      ▼
04_embeddings.py             → data/processed/chunk_embeddings.npy
      │
      ▼
05_create_vector_store.py    → vector_store/chroma_db/ (ChromaDB)
      │
      ▼
06_retrieve_context.py       → Hybrid retrieval (BM25 + Embeddings)
      │
      ▼
06b_build_context.py         → Context package (filter/dedupe/order/budget)
      │
      ▼
07_generate_answer.py        → OpenAI أو Gemini (weak/better/strict prompts)
      │
      ▼
streamlit_app.py             → الواجهة

مسار موازي (offline):
08_evaluate_retrieval.py     → evaluation/metrics_report.csv
09_log_failures.py           → evaluation/failure_log.csv
```

## 🌐 المصادر (sources.json)

كل مواقع الـ documentation اللي هيتجمعوا موجودين في `sources.json` في جذر
المشروع، كل واحد فيه:

```json
{
  "name": "nltk",
  "start_url": "https://www.nltk.org/book/ch01.html",
  "allowed_path_prefix": "https://www.nltk.org/book/",
  "max_pages": 20
}
```

- **name**: اسم المجلد اللي هيتحفظ فيه (`data/raw/<name>/`)
- **start_url**: أول صفحة يبدأ منها (الكود بيتابع أي redirect لوحده)
- **allowed_path_prefix**: أي رابط برا المسار ده هيتجاهل، حتى لو نفس الدومين
  (مهم عشان منجريش ورا الموقع كله)
- **max_pages**: حد أقصى لعدد الصفحات لكل مصدر

المصادر الجاهزة دلوقتي (14 مصدر):
| name | الموقع | ليه؟ |
|---|---|---|
| `langchain` | LangChain OSS docs | أساس الـ RAG (retrievers, chains, prompts) |
| `langgraph` | LangGraph docs | تنسيق الـ agents متعددة الخطوات |
| `huggingface_transformers` | HF Transformers | معمارية النماذج الحديثة |
| `huggingface_tokenizers` | HF Transformers (tokenizer_summary) | أساسيات الـ tokenization |
| `huggingface_tasks` | HF Transformers (tasks) | مهام NLP المدعومة |
| `faiss` | faiss.ai | Indexes, Similarity Search |
| `sentence_transformers` | sbert.net | Embeddings, Semantic Search |
| `sentence_transformers_semantic_search` | SBERT examples | تطبيق عملي على الـ semantic search |
| `scikit_learn` | scikit-learn docs | TF-IDF, Cosine Similarity |
| `pytorch` | PyTorch docs | إطار العمل الأساسي للتدريب |
| `tensorflow` | TensorFlow guide | إطار عمل بديل |
| `keras` | keras.io | API عالي المستوى فوق TensorFlow |
| `spacy` | spaCy usage docs | NLP صناعي حديث |
| `nltk` | NLTK Book | أساسيات الـ NLP الكلاسيكية |

⚠️ **ملاحظة:** `huggingface_transformers`, `huggingface_tokenizers`,
و`huggingface_tasks` عندهم نفس `allowed_path_prefix` بالظبط، يعني
احتمال كبير يجمعوا نفس الصفحات تقريبًا (تكرار). لو حصل كده، فيه سكريبت
جاهز (`dedupe_langchain.py`) بيعمل نفس الحاجة لـ LangChain — ينفع
تكيّفيه بسهولة لـ Hugging Face لو احتجتي.

💡 عايز تضيف Python docs العامة كمان (لو المشروع هيبقى AI Learning Hub
مش RAG بس)؟ ضيف العنصر ده في `sources.json`:
```json
{
  "name": "python_docs",
  "start_url": "https://docs.python.org/3/tutorial/index.html",
  "allowed_path_prefix": "https://docs.python.org/3/tutorial/",
  "max_pages": 60
}
```

### تشغيل كل المصادر مرة واحدة
```bash
python 00_data_collection.py
```

### تشغيل مصدر واحد بس (بالاسم)
```bash
python 00_data_collection.py langchain
python 00_data_collection.py scikit_learn
```

### إضافة موقع جديد
افتح `sources.json` وضيف عنصر جديد بنفس الشكل. جرب `start_url` في المتصفح
الأول، وحدد `allowed_path_prefix` بحيث يغطي القسم اللي عايزه بس (مش الموقع
كله).

## 🆓 عايز تجرب من غير ما تدفع فلوس؟ استخدم Gemini

المشروع بيدعم Google Gemini كبديل مجاني تمامًا (من غير بطاقة ائتمان):

1. خد مفتاح مجاني من **https://aistudio.google.com/apikey**
2. في `.env`:
   ```
   MODEL_PROVIDER=gemini
   GEMINI_API_KEY=المفتاح_بتاعك
   GEMINI_MODEL=gemini-flash-latest
   ```
3. جرب:
   ```bash
   python 07_generate_answer.py "How do I create an agent with tools?"
   ```

لو عايز ترجع لـ OpenAI تاني، غيّر `MODEL_PROVIDER=openai` وسيب
`OPENAI_API_KEY` متملي في `.env`.

⚠️ Gemini بيدي rate limits محدودة على الـ free tier — كفاية للتجربة
والتعلم. أسامي الموديلات بتتغيّر بمرور الوقت (Google بتقفل موديلات قديمة
للمستخدمين الجدد)، فلو `gemini-flash-latest` ما اشتغلش، جرب اسم موديل أحدث
من https://ai.google.dev/gemini-api/docs/models

## 🎨 الواجهة (بنية الملفات)

الواجهة مقسّمة لملفات منفصلة بدل ملف واحد ضخم:

| الملف | المسؤولية |
|---|---|
| `streamlit_app.py` | نقطة الدخول: يربط كل حاجة + صفحات Knowledge Base/Analytics/Settings/About |
| `sidebar.py` | الشريط الجانبي (nav + settings + فلترة الـ knowledge base) |
| `chat.py` | فقاعات المحادثة (User/AI) + `run_query()` اللي بتوصّل الـ pipeline |
| `analysis_tabs.py` | تبويبات Sources / Retrieved Chunks / Details تحت كل إجابة |
| `components.py` | HTML snippets قابلة لإعادة الاستخدام (metric card, source card...) |
| `utils_app.py` | تحميل CSS + هوية ألوان كل مصدر + دوال تنسيق (اسمه `utils_app.py` مش `utils.py` عشان منتصدمش مع `utils/` بتاع الـ pipeline) |
| `styles.css` | كل الطلاء البصري (ألوان، خطوط، animations) |

⚠️ التبويب **Details** بيعرض بيانات حقيقية (وقت الـ embedding/retrieval/
generation، عدد الـ tokens، confidence) — مش أرقام وهمية. عشان كده أضفنا
قياس وقت فعلي في `06_retrieve_context.py` (`self.last_timing`) وقراءة
`usage_metadata`/`usage` من رد الـ API في `07_generate_answer.py`. التغييرات
دي **إضافية بالكامل** (backward-compatible) — أي كود قديم بينادي
`retriever.retrieve()` أو `generate_answer()` لسه شغال زي ما هو.

## 🚀 النشر على الإنترنت (Streamlit Community Cloud)

الطريقة دي مجانية بالكامل وبتديكي لينك عام (`https://xxxx.streamlit.app`)
يفتح الواجهة من غير ما حد يشغّل حاجة على جهازه.

### الخطوة 1 — تأكدي إن حجم البيانات معقول
```powershell
Get-ChildItem -Recurse vector_store, data\processed | Measure-Object -Property Length -Sum
```
لو الحجم الكلي أكبر من ~300-400 ميجا، فكري تقلّلي `max_pages` في `sources.json`
وتعيدي بناء الـ pipeline (GitHub بيرفض أي ملف أكبر من 100 ميجا لوحده، ومستودعات
كبيرة جدًا بتبقى بطيئة).

### الخطوة 2 — ارفعي المشروع على GitHub
```powershell
git init
git add .
git commit -m "AI Learning Hub"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git push -u origin main
```
(لو مش عندك repo لسه، اعمليه فاضي من على github.com الأول بدون README
عشان الأمر ده يشتغل عادي)

⚠️ `.env` مش هيترفع (متجاهل من `.gitignore` عمدًا) — وده مقصود، عشان
متسربيش مفتاح الـ API بتاعك على GitHub.

### الخطوة 3 — نشر الواجهة
1. روحي **https://share.streamlit.io** وسجّلي دخول بحساب GitHub بتاعك
2. دوسي **"New app"**
3. اختاري الـ repo، والـ branch (`main`)، والملف الرئيسي: `streamlit_app.py`
4. قبل ما تدوسي Deploy، افتحي **"Advanced settings" → "Secrets"** والصقي فيها:
   ```toml
   MODEL_PROVIDER = "gemini"
   GEMINI_API_KEY = "المفتاح الحقيقي بتاعك"
   GEMINI_MODEL = "gemini-flash-latest"
   ```
5. دوسي **Deploy** واستني كذا دقيقة لحد ما يثبّت المكتبات ويشتغل.

الكود بقى بيدور على المفتاح في الـ environment variables الأول، ولو
مالقاهوش بيدوّر في Streamlit Secrets تلقائيًا — مفيش تعديل تاني مطلوب.

### تحديث الواجهة بعد النشر
أي `git push` جديد على `main` بيعمل إعادة نشر تلقائي للواجهة على نفس اللينك.

## ⚙️ التجهيز (تشغيل محلي)

```bash
# 1. بيئة افتراضية (اختياري بس مستحسن)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. تثبيت المكتبات
pip install -r requirements.txt

# 3. إعداد متغيرات البيئة
cp .env.example .env
# افتح .env واختار MODEL_PROVIDER (gemini أو openai) واملأ المفتاح المناسب
```

## 🚀 التشغيل (أول مرة، بالترتيب)

```bash
python 00_data_collection.py     # كل المصادر في sources.json
python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 04_embeddings.py
python 05_create_vector_store.py

# جرب من التيرمينال قبل ما تفتح الواجهة
python 06_retrieve_context.py "What is BM25?"
python 07_generate_answer.py "What is BM25?"

# الواجهة
streamlit run streamlit_app.py
```

## 📊 التقييم وتتبع الأخطاء

```bash
python 08_evaluate_retrieval.py   # يقارن BM25 vs Embeddings vs Hybrid بالأرقام
python 09_log_failures.py         # يسجل أي سؤال فشلت فيه أي طبقة
```

⚠️ **مهم:** ملف `evaluation/eval_queries.json` فيه أسئلة تجريبية. لازم بعد
ما تشغّل `00_data_collection.py` وتشوف أسماء الملفات الفعلية اللي اتنزلت في
`data/raw/<source>/`، ترجع تعدّل `eval_queries.json` بعناوين حقيقية موجودة
فعلاً عندك، وإلا أرقام الـ evaluation هتكون غير موثوقة.

**ملاحظة عن matching الأسئلة:** العنوان بيتولد من اسم الملف بعد ما `_`
تتحول لمسافة (يعني `oss_python_langchain_agents.txt` → `"oss python
langchain agents"`)، فحط `relevant_titles` بمسافات مش underscores.

## 🧠 إعدادات مهمة في `.env`

| المتغير | الوظيفة |
|---|---|
| `MODEL_PROVIDER` | `openai` أو `gemini` |
| `HYBRID_ALPHA` | وزن الـ semantic في الـ hybrid retrieval (0=BM25 بس, 1=embeddings بس) |
| `TOP_K` | عدد الـ chunks اللي بترجع من الـ retrieval |
| `CHROMA_COLLECTION_NAME` | اسم الـ collection في ChromaDB |

## 📂 هيكل المشروع

```
AI-Learning-Hub/
├── sources.json                (ليستة كل مواقع الـ documentation)
├── data/
│   ├── raw/<source_name>/*.txt
│   ├── processed/
│   └── metadata/
├── vector_store/chroma_db/
├── evaluation/
│   ├── eval_queries.json
│   ├── metrics_report.csv     (بيتولد بعد 08)
│   └── failure_log.csv        (بيتولد بعد 09)
├── utils/
│   ├── config.py               (كل الإعدادات من .env)
│   └── logging_utils.py        (logger + log_failure)
├── 00_data_collection.py … 09_log_failures.py
├── streamlit_app.py
├── requirements.txt
├── .env.example
└── README.md
```

## ⚠️ ملاحظة عن مواقع بتعمل redirect (زي LangChain)

بعض المواقع (زي LangChain اللي نقلت الـ docs لـ `docs.langchain.com`)
بتعمل **redirect** للصفحة القديمة. الكود بيتابع الـ redirect ده تلقائيًا
ويستخدم `allowed_path_prefix` بتاع كل مصدر في `sources.json` عشان يحدد
نطاق الجمع الصح على الموقع الجديد.

لو حسيت إن `00_data_collection.py` رجعلك عدد صفحات قليل جدًا أو صفر لمصدر
معين:
1. تأكد إن `start_url` بتاعه في `sources.json` صفحة شغالة فعلًا.
2. لاحظ في اللوج سطر "الرابط اتعمله redirect: ... → ..." — ده بيوريك
   الرابط الحقيقي اللي الموقع رجّعه، استخدمه كمرجع تظبط عليه
   `allowed_path_prefix`.

## 🔁 لو غيّرت مصادر الـ documentation

- عدّل `sources.json` (ضيف/شيل/غيّر مصدر)
- شيل محتوى `data/raw/`, `data/processed/`, `vector_store/chroma_db/`
- شغّل الـ pipeline من الأول (00 → 05)
