import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

UA = "genre-map-proto/1.0 (personal project)"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WDQS = "https://query.wikidata.org/sparql"
WIKI_API = "https://en.wikipedia.org/w/api.php"

MUSIC_GENRE_QID = "Q188451"  # "music genre" item in Wikidata

# Very conservative filter to reject place pages in Wikipedia fallback
MUSIC_HINTS = ["music", "musical", "genre", "style", "band", "song", "album"]
PLACE_HINTS = ["city", "town", "county", "province", "state", "capital", "population", "located in"]


def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_overrides(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = norm(row.get("genre", ""))
            t = (row.get("wiki_title") or "").strip()
            if g and t:
                out[g] = t
    return out


def http_get(url: str, params: dict, timeout: int = 60) -> dict:
    """
    GET with a polite User-Agent, and basic 429 handling.
    """
    headers = {"User-Agent": UA}
    for attempt in range(6):
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            sleep_s = int(retry) if retry and retry.isdigit() else (2 + attempt * 2)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Too many requests repeatedly for {url}")


def wdqs_ask_is_music_genre(qid: str) -> bool:
    """
    Ask WDQS: is this item an instance/subclass (transitively) of music genre?
    Tiny queries like this are much more reliable than one huge "download everything" query.
    """
    query = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    ASK {{
      {{
        wd:{qid} wdt:P31/wdt:P279* wd:{MUSIC_GENRE_QID} .
      }} UNION {{
        wd:{qid} wdt:P279* wd:{MUSIC_GENRE_QID} .
      }}
    }}
    """
    headers = {
        "User-Agent": UA,
        "Accept": "application/sparql-results+json",
    }

    for attempt in range(6):
        r = requests.get(WDQS, params={"query": query, "format": "json"}, headers=headers, timeout=60)
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            sleep_s = int(retry) if retry and retry.isdigit() else (2 + attempt * 2)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
        js = r.json()
        return bool(js.get("boolean", False))
    return False


def wikidata_search_candidates(genre: str, limit: int = 8) -> List[Tuple[str, str]]:
    """
    Returns [(qid, label), ...] from Wikidata's search.
    """
    js = http_get(
        WIKIDATA_API,
        params={
            "action": "wbsearchentities",
            "search": genre,
            "language": "en",
            "format": "json",
            "limit": limit,
        },
        timeout=30,
    )
    out = []
    for hit in js.get("search", []):
        qid = hit.get("id")
        label = hit.get("label") or ""
        if qid and qid.startswith("Q"):
            out.append((qid, label))
    return out


def wikidata_best_music_genre_qid(genre: str) -> Optional[Tuple[str, str]]:
    """
    Find the best Wikidata item for this genre that actually is a music genre.
    Returns (qid, best_label) or None.
    """
    # Search terms that push results toward music-genre meanings
    search_terms = [
        genre,
        f"{genre} music genre",
        f"{genre} genre",
        f"{genre} musical style",
    ]

    seen = set()
    candidates: List[Tuple[str, str]] = []

    for term in search_terms:
        for qid, label in wikidata_search_candidates(term, limit=8):
            if qid not in seen:
                seen.add(qid)
                candidates.append((qid, label))

    # Validate with WDQS ASK (this avoids cities/people/etc.)
    for qid, label in candidates:
        try:
            if wdqs_ask_is_music_genre(qid):
                return qid, (label or qid)
        except Exception:
            continue

    return None


def wikipedia_intro(title: str) -> Optional[str]:
    js = http_get(
        WIKI_API,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
        },
        timeout=30,
    )
    pages = js.get("query", {}).get("pages", {})
    page = next(iter(pages.values())) if pages else {}
    extract = (page.get("extract") or "").strip()
    if not extract:
        return None
    return extract.split("\n")[0].strip()


def wikipedia_intro_looks_like_music(intro: Optional[str]) -> bool:
    if not intro:
        return False
    p = intro.lower()
    if not any(t in p for t in MUSIC_HINTS):
        return False
    if any(t in p for t in PLACE_HINTS) and ("music" not in p and "musical" not in p):
        return False
    return True


def wikipedia_title_from_search(genre: str) -> Optional[str]:
    """
    Strict fallback: only accept pages whose intro looks music-related.
    """
    queries = [
        f'intitle:"{genre}" (music OR genre OR style)',
        f'"{genre}" (music OR genre OR style)',
        f"{genre} music genre",
        f"{genre} musical style",
    ]

    for q in queries:
        try:
            js = http_get(
                WIKI_API,
                params={"action": "query", "list": "search", "srsearch": q, "srlimit": 6, "format": "json"},
                timeout=30,
            )
            for hit in js.get("query", {}).get("search", []):
                title = hit.get("title")
                if not title:
                    continue
                intro = wikipedia_intro(title)
                if wikipedia_intro_looks_like_music(intro):
                    return title
            time.sleep(0.2)
        except Exception:
            continue

    return None


def wikidata_enwiki_title_for_qid(qid: str) -> Optional[str]:
    """
    Get the English Wikipedia sitelink title for a Wikidata item.
    """
    js = http_get(
        WIKIDATA_API,
        params={
            "action": "wbgetentities",
            "ids": qid,
            "props": "sitelinks",
            "sitefilter": "enwiki",
            "format": "json",
        },
        timeout=30,
    )
    ent = js.get("entities", {}).get(qid, {})
    sl = ent.get("sitelinks", {}).get("enwiki", {})
    title = sl.get("title")
    return title


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    input_csv = data_dir / "genre_attrs.csv"
    if not input_csv.exists():
        raise SystemExit("Missing data/genre_attrs.csv (you said you added it — double-check the path).")

    overrides_path = data_dir / "wiki_overrides.csv"
    out_path = data_dir / "wiki_map.csv"
    needs_review_path = data_dir / "wiki_needs_review.csv"

    overrides = load_overrides(overrides_path)

    df = pd.read_csv(input_csv)
    if "genre" not in df.columns:
        raise SystemExit("Input CSV must contain a 'genre' column.")

    rows_out = []
    rows_review = []

    total = len(df)
    matched = 0
    matched_wikidata = 0
    matched_wiki_fallback = 0

    for i, raw_genre in enumerate(df["genre"].astype(str).tolist(), start=1):
        g = raw_genre.strip()
        ng = norm(g)

        if i % 200 == 0:
            print(f"... {i}/{total} processed (matched {matched})")

        # 1) manual override wins
        if ng in overrides:
            rows_out.append({"genre": g, "wiki_title": overrides[ng], "confidence": "high", "source": "override"})
            matched += 1
            continue

        # 2) Wikidata (validated)
        best = wikidata_best_music_genre_qid(g)
        if best:
            qid, _label = best
            title = wikidata_enwiki_title_for_qid(qid)
            if title:
                rows_out.append({"genre": g, "wiki_title": title, "confidence": "high", "source": "wikidata"})
                matched += 1
                matched_wikidata += 1
                continue

        # 3) Wikipedia strict fallback
        title = wikipedia_title_from_search(g)
        if title:
            rows_out.append({"genre": g, "wiki_title": title, "confidence": "medium", "source": "wikipedia_fallback"})
            matched += 1
            matched_wiki_fallback += 1
        else:
            rows_out.append({"genre": g, "wiki_title": "", "confidence": "none", "source": "unmatched"})
            rows_review.append({"genre": g})

        time.sleep(0.05)  # small politeness delay

    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    pd.DataFrame(rows_review).to_csv(needs_review_path, index=False)

    print("Done.")
    print(f"Matched: {matched}/{total}")
    print(f"  via Wikidata: {matched_wikidata}")
    print(f"  via Wikipedia fallback: {matched_wiki_fallback}")
    print(f"Wrote: {out_path}")
    print(f"Needs review: {needs_review_path} (unmatched genres)")

if __name__ == "__main__":
    main()
