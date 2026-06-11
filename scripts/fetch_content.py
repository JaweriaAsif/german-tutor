"""One-time fetcher for open lesson content.

Pulls REAL German lessons + grammar from the English Wikibooks "German" course
(MediaWiki API, CC BY-SA) and a public-domain reading text from Project Gutenberg
(Gutendex), and writes them under `content/` with an `index.json` that the
`get_lesson_material` tool reads. Run once (committed output), NOT per request.

    uv run python scripts/fetch_content.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CONTENT = Path(__file__).resolve().parents[1] / "content"
WIKIBOOKS_API = "https://en.wikibooks.org/w/api.php"
GUTENDEX = "https://gutendex.com/books?languages=de"
UA = "german-tutor-content-fetch/1.0 (educational; open content)"
MAX_PAGE_CHARS = 9000
MAX_READING_CHARS = 30000

# (slug, Wikibooks page title) — real pages confirmed to exist in the course.
LESSON_PAGES = [
    ("lesson-01", "German/Lesson 1"),
    ("lesson-02", "German/Lesson 2"),
    ("lesson-03", "German/Lesson 3"),
    ("lesson-04", "German/Lesson 4"),
    ("lesson-05", "German/Lesson 5"),
    ("lesson-06", "German/Lesson 6"),
    ("lesson-07", "German/Lesson 7"),
    ("lesson-08", "German/Lesson 8"),
    ("level1-essen", "German/Level I/Essen"),
    ("level1-freizeit", "German/Level I/Freizeit"),
    ("level1-kleidung", "German/Level I/Kleidung"),
    ("level1-familie", "German/Level I/Volk und Familie"),
]
GRAMMAR_PAGES = [
    ("nouns", "German/Grammar/Nouns"),
    ("cases", "German/Grammar/Cases"),
    ("verbs", "German/Grammar/Verbs"),
    ("pronouns", "German/Grammar/Pronouns"),
    ("sentences", "German/Grammar/Sentences"),
    ("dative-prepositions", "German/Grammar/Dative prepositions"),
    ("alphabet", "German/Grammar/Alphabet and Pronunciation"),
]


def _get(url: str) -> bytes:
    # Be polite to Wikibooks: throttle + back off on 429.
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                time.sleep(1.0)
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def _wikibooks_extract(title: str) -> str:
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": "1",
        "redirects": "1", "format": "json", "titles": title,
    })
    data = json.loads(_get(f"{WIKIBOOKS_API}?{params}"))
    page = next(iter(data["query"]["pages"].values()))
    return (page.get("extract") or "").strip(), page.get("title", title)


def fetch_pages(pages, kind: str) -> list[dict]:
    out = []
    (CONTENT / kind).mkdir(parents=True, exist_ok=True)
    for slug, title in pages:
        try:
            text, real_title = _wikibooks_extract(title)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {kind}/{slug} '{title}' failed: {e}")
            continue
        if not text:
            print(f"  - {kind}/{slug} '{title}' empty/missing, skipped")
            continue
        text = text[:MAX_PAGE_CHARS]
        path = CONTENT / kind / f"{slug}.txt"
        path.write_text(text, encoding="utf-8")
        out.append({
            "id": f"{kind}/{slug}", "type": kind, "title": real_title,
            "source": f"English Wikibooks: {title}", "license": "CC BY-SA 3.0",
            "path": str(path.relative_to(CONTENT.parent)),
        })
        print(f"  + {kind}/{slug} ({len(text)} chars)")
    return out


def fetch_reading() -> list[dict]:
    (CONTENT / "reading").mkdir(parents=True, exist_ok=True)
    try:
        catalog = json.loads(_get(GUTENDEX))
    except Exception as e:  # noqa: BLE001
        print(f"  ! gutendex failed: {e}")
        return []
    for book in catalog.get("results", []):
        txt_url = next((u for fmt, u in book.get("formats", {}).items()
                        if fmt.startswith("text/plain")), None)
        if not txt_url:
            continue
        try:
            raw = _get(txt_url).decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            print(f"  ! download failed: {e}")
            continue
        slug = "".join(c if c.isalnum() else "-" for c in book["title"].lower())[:40].strip("-")
        path = CONTENT / "reading" / f"{slug}.txt"
        path.write_text(raw[:MAX_READING_CHARS], encoding="utf-8")
        author = ", ".join(a["name"] for a in book.get("authors", [])) or "unknown"
        print(f"  + reading/{slug} — {book['title']} ({author})")
        return [{
            "id": f"reading/{slug}", "type": "reading", "title": book["title"],
            "source": f"Project Gutenberg #{book['id']} — {author}",
            "license": "Public domain", "path": str(path.relative_to(CONTENT.parent)),
        }]
    return []


def main() -> int:
    CONTENT.mkdir(parents=True, exist_ok=True)
    print("Fetching lessons (Wikibooks)…")
    items = fetch_pages(LESSON_PAGES, "lessons")
    print("Fetching grammar (Wikibooks)…")
    items += fetch_pages(GRAMMAR_PAGES, "grammar")
    print("Fetching reading (Project Gutenberg)…")
    items += fetch_reading()
    (CONTENT / "index.json").write_text(
        json.dumps({"schema": "german_tutor.content.v1", "items": items},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(items)} item(s) to {CONTENT}/index.json")
    return 0 if items else 1


if __name__ == "__main__":
    raise SystemExit(main())
