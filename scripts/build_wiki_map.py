import csv
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

UA = "genre-map-proto/1.0 (personal project)"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WDQS = "https://query.wikidata.org/sparql"
WIKI_API = "https://en.wikipedia.org/w/api.php"

MUSIC_GENRE_QID = "Q188451"  # music genre

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


def http_get(session: requests.Session, url: str, params: dict, timeout: int = 60) -> dict:
    headers = {"User-Agent": UA}
    for attempt in range(6):
        r = session.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            sleep_s = int(retry) if retry and retry.isdigit() else (2 + attempt * 2)
            print(f"Rate limited (429). Sleeping {sleep_s}s...", flush=True)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Too many requests repeatedly for {url}")


def wdqs_ask_is_music_genre(session: requests.Session, qid: str) -> bool:
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
    headers = {"User-Agent": UA, "Accept": "application/sparql-results+json"}
    for attempt in range(6):
        r = session.get(WDQS, params={"query": query, "format": "json"}, headers=headers, timeout=60)
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            sleep_s = int(retry) if retry and retry.isdigit() else (2 + attempt * 2)
            print(f"WDQS 429. Sleeping {sleep_s}s...", flush=True)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
        return bool(r.json().get("boolean", False))
    return False


def wikidata_search_candidates(session: requests.Session, query_text: str, limit: int = 6) -> List[Tuple[str, str]]:
    js = http_get(
        session,
        WIKIDATA_API,
        params={
            "action": "wbsearchentities",
            "search": query_text,
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


def wikidata_enwiki_title_for_qid(session: requests.Session, qid: str) -> Optional[str]:
    js = http_get(
        session,
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
    return sl.get("title")


def wikipedia_intro(session: requests.Session, title: str) -> Optional[str]:
    js = http_get(
        session,
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


def wikipedia_title_from_search(session: requests.Session, genre: str) -> Optional[str]:
    queries = [
        f'intitle:"{genre}" (music OR genre OR style)',
        f'"{genre}" (music OR genre OR style)',
        f"{genre} music genre",
        f"{genre} musical style",
    ]
    for q in queries:
        try:
            js = http_get(
                session,
                WIKI_API,
                params={"action": "query", "list": "search", "srsearch": q, "srlimit": 6, "format": "json"},
                timeout=30,
            )
            for hit in js.get("query", {}).get("search", []):
                title = hit.get("title")
                if not title:
                    continue
                intro = wikipedia_intro(session, title)
                if wikipedia_intro_looks_like_music(intro):
                    return title
            time.sleep(0.2)
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Process only first N genres (test). Use 0 for all.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between genres.")
    args = parser.parse_args()

    print("Starting build_wiki_map.py ...", flush=True)

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    input_csv = data_dir / "genre_attrs.csv"

    if not input_csv.exists():
        raise SystemExit("Missing data/genre_attrs.csv. Make sure it's in the data/ folder.")

    overrides_path = data_dir / "wiki_overrides.csv"
    out_path = data_dir / "wiki_map.csv"
    needs_review_path = data_dir / "wiki_needs_review.csv"

    overrides = load_overrides(overrides_path)

    df = pd.read_csv(input_csv)
    if "genre" not in df.columns:
        raise SystemExit("Input CSV must contain a 'genre' column.")

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    total = len(df)
    print(f"Loaded {total} genres (limit={args.limit}).", flush=True)

    session = requests.Session()

    rows_out = []
    rows_review = []

    matched = 0
    matched_wikidata = 0
    matched_wiki_fallback = 0

    qid_cache: Dict[str, bool] = {}

    for i, raw_genre in enumerate(df["genre"].astype(str).tolist(), start=1):
        g = raw_genre.strip()
        ng = norm(g)

        if i == 1 or i % 10 == 0:
            print(f"... {i}/{total} processing: {g}", flush=True)

        # override wins
        if ng in overrides:
            rows_out.append({"genre": g, "wiki_title": overrides[ng], "confidence": "high", "source": "override"})
            matched += 1
            continue

        # Wikidata: search candidates and validate "is music genre"
        wiki_title = None
        try:
            candidates = wikidata_search_candidates(session, f"{g} music genre", limit=6)
            candidates += wikidata_search_candidates(session, g, limit=6)
        except Exception:
            candidates = []

        seen = set()
        for qid, _label in candidates:
            if qid in seen:
                continue
            seen.add(qid)

            if qid not in qid_cache:
                try:
                    qid_cache[qid] = wdqs_ask_is_music_genre(session, qid)
                except Exception:
                    qid_cache[qid] = False

            if qid_cache[qid]:
                t = wikidata_enwiki_title_for_qid(session, qid)
                if t:
                    wiki_title = t
                    break

        if wiki_title:
            rows_out.append({"genre": g, "wiki_title": wiki_title, "confidence": "high", "source": "wikidata"})
            matched += 1
            matched_wikidata += 1
            time.sleep(args.sleep)
            continue

        # Wikipedia strict fallback
        title = wikipedia_title_from_search(session, g)
        if title:
            rows_out.append({"genre": g, "wiki_title": title, "confidence": "medium", "source": "wikipedia_fallback"})
            matched += 1
            matched_wiki_fallback += 1
        else:
            rows_out.append({"genre": g, "wiki_title": "", "confidence": "none", "source": "unmatched"})
            rows_review.append({"genre": g})

        time.sleep(args.sleep)

    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    pd.DataFrame(rows_review).to_csv(needs_review_path, index=False)

    print("Done.", flush=True)
    print(f"Matched: {matched}/{total}", flush=True)
    print(f"  via Wikidata: {matched_wikidata}", flush=True)
    print(f"  via Wikipedia fallback: {matched_wiki_fallback}", flush=True)
    print(f"Wrote: {out_path}", flush=True)
    print(f"Needs review: {needs_review_path}", flush=True)


if __name__ == "__main__":
    main()
