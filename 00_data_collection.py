"""
00_data_collection.py
----------------------
Phase 0 — Data Collection (Multi-Source)

بيقرأ ليستة مواقع من sources.json، وبيجمع كل موقع لوحده:
- بيدخل على start_url بتاعه (ويتابع أي redirect لوحده)
- يجمع كل الروابط الداخلية اللي بادئة بـ allowed_path_prefix
- يزور كل صفحة، يستخرج المحتوى الرئيسي بس
- يحفظها في data/raw/<source_name>/

تشغيل (كل المواقع في sources.json):
    python 00_data_collection.py

تشغيل موقع واحد بس (بالاسم اللي في sources.json):
    python 00_data_collection.py langchain
"""

import json
import sys
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils.config import DATA_RAW_DIR, SOURCES_CONFIG_PATH
from utils.logging_utils import get_logger

logger = get_logger("00_data_collection")

HEADERS = {
    # بعض المواقع (زي spacy.io) بترفض أي User-Agent بيعرّف نفسه صراحة إنه
    # bot. الشكل ده بيقلد متصفح حقيقي عشان الطلبات تعدي عادي.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_DELAY_SECONDS = 1.0  # نكون مؤدبين مع السيرفر (زودتها من 0.5 لتقليل الحظر)
MAX_RETRIES_ON_429 = 4


def fetch_with_retry(url: str, timeout: int = 15) -> requests.Response:
    """
    زي requests.get() عادي، لكن لو السيرفر رجّع 429 (Too Many Requests -
    زي ما حصل مع huggingface.co) بننتظر ونحاول تاني بدل ما نستسلم على طول.
    المدة بتزيد كل محاولة (exponential backoff)، ونحترم هيدر Retry-After
    لو السيرفر بعته صراحةً.
    """
    resp = None
    for attempt in range(1, MAX_RETRIES_ON_429 + 1):
        resp = requests.get(url, headers=HEADERS, timeout=timeout)

        if resp.status_code != 429:
            resp.raise_for_status()
            return resp

        retry_after = resp.headers.get("Retry-After", "")
        if retry_after.strip().isdigit():
            wait = float(retry_after)
        else:
            wait = min(REQUEST_DELAY_SECONDS * (2**attempt), 30)

        if attempt < MAX_RETRIES_ON_429:
            logger.warning(
                f"429 Too Many Requests على {url} — بنستنى {wait:.0f} ثانية "
                f"ونحاول تاني (محاولة {attempt}/{MAX_RETRIES_ON_429})"
            )
            time.sleep(wait)

    resp.raise_for_status()  # لو خلصت المحاولات ولسه 429، نطلع الخطأ عادي
    return resp

# tags/classes بنشيلها لأنها navigation/footer/ads/sidebar مش محتوى حقيقي.
# القايمة دي بتغطي Docusaurus/Mintlify (زي langchain) و Sphinx/pydata-theme
# (زي scikit-learn, pytorch) و صفحات Hugging Face.
NOISE_SELECTORS = [
    "nav", "footer", "header", "script", "style", "button", "form",
    "[role='navigation']",
    # Docusaurus / Mintlify
    ".theme-doc-sidebar-container", ".theme-doc-toc-desktop",
    ".navbar", ".footer", ".breadcrumbs",
    # Sphinx / pydata-sphinx-theme (scikit-learn, pytorch)
    ".bd-sidebar-primary", ".bd-sidebar-secondary", ".bd-header",
    ".bd-footer", ".sphinxsidebar", "#searchbox", ".related",
    ".pytorch-left-menu", ".pytorch-right-menu", ".pytorch-breadcrumbs-wrapper",
    # عام
    "#main-content nav", ".skip-link",
]

# ترتيب البحث عن المحتوى الرئيسي: أول selector يلاقي حاجة فيه، بنستخدمه.
# الأكتر تحديدًا (specific) الأول، عشان لو الصفحة فيها <main> واسع بيلف
# السايدبار كمان، نمسك المحتوى الحقيقي بس مش السايدبار.
MAIN_CONTENT_SELECTORS = [
    ".markdown-body",  # GitHub wiki (زي faiss)
    ".bd-article", "#main-content",
    "main", "article", "[role='main']",
    ".document", ".body",
]


def load_sources() -> list[dict]:
    if not SOURCES_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{SOURCES_CONFIG_PATH} مش موجود. اعمل ملف sources.json فيه ليستة "
            "المواقع (name, start_url, allowed_path_prefix, max_pages)."
        )
    with open(SOURCES_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def is_same_domain(base_url: str, candidate_url: str) -> bool:
    return urlparse(base_url).netloc == urlparse(candidate_url).netloc


def resolve_start_url(url: str) -> str:
    """
    مواقع كتير بتعمل 301/302 redirect للصفحة اللي انت طالبها (زي
    docs.langchain.com الجديد). الدالة دي بتعمل request واحد بس عشان تعرف
    الـ URL الحقيقي بعد أي redirect، ونستخدمه هو كأساس لكل المقارنات.
    """
    try:
        resp = fetch_with_retry(url)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status == 403:
            logger.error(
                f"الموقع رفض الطلب (403 Forbidden) على {url}. "
                "غالبًا بيحظر الـ scraping. جرب تفتح الرابط في متصفح عادي وتتأكد."
            )
        else:
            logger.error(f"فشل تحميل {url} (HTTP {status}): {e}")
        raise
    if resp.url != url:
        logger.info(f"الرابط اتعمله redirect: {url} → {resp.url}")
    return resp.url


def collect_internal_links(
    start_url: str, max_pages: int, path_prefix: str, delay: float = REQUEST_DELAY_SECONDS
) -> list[str]:
    """BFS بسيط يجمع لينكات داخلية بادئة بـ path_prefix."""
    to_visit = [start_url]
    seen = set()

    while to_visit and len(seen) < max_pages:
        url = to_visit.pop(0)
        if url in seen:
            continue
        seen.add(url)

        try:
            resp = fetch_with_retry(url)
        except requests.RequestException as e:
            logger.warning(f"تعذر تحميل {url}: {e}")
            continue

        final_url = resp.url  # الرابط الحقيقي بعد أي redirect

        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.find_all("a", href=True):
            link = urljoin(final_url, a["href"]).split("#")[0]
            if (
                is_same_domain(path_prefix, link)
                and link.startswith(path_prefix)
                and link not in seen
                and link not in to_visit
            ):
                to_visit.append(link)

        time.sleep(delay)

    return list(seen)


def extract_main_content(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    main = None
    for selector in MAIN_CONTENT_SELECTORS:
        main = soup.select_one(selector)
        if main is not None:
            break
    if main is None:
        main = soup.body
    if main is None:
        return ""

    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    slug = path.replace("/", "_") or "index"
    return slug


def scrape_source(source: dict) -> int:
    """بيجمع موقع واحد بالكامل، ويرجّع عدد الصفحات اللي اتحفظت."""
    name = source["name"]
    out_dir = DATA_RAW_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== المصدر: {name} ===")
    try:
        start_url = resolve_start_url(source["start_url"])
    except requests.RequestException as e:
        logger.error(f"مقدرش أوصل لـ {source['start_url']}: {e}")
        return 0

    path_prefix = source.get("allowed_path_prefix") or start_url
    max_pages = source.get("max_pages", 150)
    delay = source.get("request_delay", REQUEST_DELAY_SECONDS)

    logger.info(
        f"بنجمع الروابط بادئين من {start_url} (نطاق: {path_prefix}, حد أقصى {max_pages} صفحة)..."
    )
    urls = collect_internal_links(start_url, max_pages, path_prefix, delay=delay)
    logger.info(f"لقينا {len(urls)} رابط. هنزورهم ونستخرج المحتوى.")

    saved = 0
    for url in tqdm(urls, desc=f"Scraping {name}"):
        try:
            resp = fetch_with_retry(url)
        except requests.RequestException as e:
            logger.warning(f"تعذر تحميل {url}: {e}")
            continue

        content = extract_main_content(resp.text)
        if len(content) < 200:  # صفحة فاضية أو شبه فاضية، متستهلش
            continue

        slug = slug_from_url(url)
        out_path = out_dir / f"{slug}.txt"
        out_path.write_text(f"SOURCE_URL: {url}\n\n{content}", encoding="utf-8")
        saved += 1
        time.sleep(delay)

    logger.info(f"تم حفظ {saved} صفحة من {name} في {out_dir}")
    return saved


def main():
    sources = load_sources()

    # لو المستخدم بعت اسم مصدر معين في الـ command line، نجمعه هو بس
    requested_name = sys.argv[1] if len(sys.argv) > 1 else None
    if requested_name:
        sources = [s for s in sources if s["name"] == requested_name]
        if not sources:
            logger.error(f"مفيش مصدر اسمه '{requested_name}' في sources.json")
            return

    total_saved = 0
    for source in sources:
        total_saved += scrape_source(source)

    logger.info(f"خلصنا. إجمالي الصفحات المحفوظة من كل المصادر: {total_saved}")


if __name__ == "__main__":
    main()
