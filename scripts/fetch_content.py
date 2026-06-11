"""One-time fetcher for open lesson content.

Pulls a few German grammar pages from English Wikibooks (MediaWiki API, CC BY-SA)
and a public-domain German reading text from Project Gutenberg (via Gutendex), and
writes them under `content/` with an `index.json` the `get_lesson_material` tool
reads. Run once (committed output), NOT per request — keeps the tutor deterministic
and offline-safe.

    uv run python scripts/fetch_content.py
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

CONTENT = Path(__file__).resolve().parents[1] / "content"
WIKIBOOKS_API = "https://en.wikibooks.org/w/api.php"
GUTENDEX = "https://gutendex.com/books?languages=de"

# (slug, Wikibooks page title) — grammar references in English about German.
GRAMMAR_PAGES = [
    ("articles", "German/Grammar/Articles"),
    ("cases", "German/Grammar/Cases"),
    ("verbs", "German/Grammar/Verbs"),
    ("pronouns", "German/Grammar/Pronouns"),
    ("alphabet", "German/Grammar/Alphabet and pronunciation"),
]
MAX_GRAMMAR_CHARS = 8000
MAX_READING_CHARS = 30000
UA = "german-tutor-content-fetch/1.0 (educational; open content)"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def fetch_grammar() -> list[dict]:
    out = []
    (CONTENT / "grammar").mkdir(parents=True, exist_ok=True)
    for slug, title in GRAMMAR_PAGES:
        params = urllib.parse.urlencode(
            {
                "action": "query", "prop": "extracts", "explaintext": "1",
                "redirects": "1", "format": "json", "titles": title,
            }
        )
        try:
            data = json.loads(_get(f"{WIKIBOOKS_API}?{params}"))
            page = next(iter(data["query"]["pages"].values()))
            text = (page.get("extract") or "").strip()
        except Exception as e:  # noqa: BLE001
            print(f"  ! grammar '{title}' failed: {e}")
            continue
        if not text:
            print(f"  - grammar '{title}' empty/missing, skipped")
            continue
        text = text[:MAX_GRAMMAR_CHARS]
        path = CONTENT / "grammar" / f"{slug}.txt"
        path.write_text(text, encoding="utf-8")
        out.append({
            "id": f"grammar/{slug}", "type": "grammar", "title": page.get("title", title),
            "source": f"English Wikibooks: {title}",
            "license": "CC BY-SA 3.0",
            "path": str(path.relative_to(CONTENT.parent)),
        })
        print(f"  + grammar/{slug} ({len(text)} chars)")
    return out


def fetch_reading() -> list[dict]:
    (CONTENT / "reading").mkdir(parents=True, exist_ok=True)
    try:
        catalog = json.loads(_get(GUTENDEX))
    except Exception as e:  # noqa: BLE001
        print(f"  ! gutendex failed: {e}")
        return []
    for book in catalog.get("results", []):
        txt_url = next(
            (u for fmt, u in book.get("formats", {}).items()
             if fmt.startswith("text/plain")), None
        )
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
        title = book["title"]
        author = ", ".join(a["name"] for a in book.get("authors", [])) or "unknown"
        print(f"  + reading/{slug} — {title} ({author})")
        return [{
            "id": f"reading/{slug}", "type": "reading", "title": title,
            "source": f"Project Gutenberg #{book['id']} — {author}",
            "license": "Public domain",
            "path": str(path.relative_to(CONTENT.parent)),
        }]
    return []


def main() -> int:
    CONTENT.mkdir(parents=True, exist_ok=True)
    print("Fetching grammar (Wikibooks)…")
    items = fetch_grammar()
    print("Fetching reading (Project Gutenberg)…")
    items += fetch_reading()
    index = {"schema": "german_tutor.content.v1", "items": items}
    (CONTENT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {len(items)} item(s) to {CONTENT}/index.json")
    return 0 if items else 1


if __name__ == "__main__":
    raise SystemExit(main())
