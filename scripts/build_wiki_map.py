import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests

WDQS = "https://query.wikidata.org/sparql"  # official endpoint alias
UA = "genre-map-proto/1.0 (personal project)"

# Normalize strings so "Detroit Rock" and "detroit rock" match
def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_overrides(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = norm(row.get("genre", ""))
            t = (row.get("wiki_title") or "").strip()
            if g and t:
                out[g] = t
    return out

def wdqs_query(query: str) -> List[Dict[str, str]]:
    headers = {
        "User-Agent": UA,
        "Accept": "application/sparql-results+json",
    }
    r = requests.get(WDQS, params={"query": query, "format": "json"}, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    bindings = data.get("results", {}).get("bindings", [])
    rows = []
    for b in bindings:
        row = {}
        for k, v in b.items():
            row[k] = v.get("value")
        rows.append(row)
    return rows

def fetch_wikidata_music_genres() -> Dict[str, Tuple[str, str, List[str]]]:
    """
    Returns dict:
      key: normalized label/alias
      val: (wiki_title, wikipedia_url, aliases[])
    We only pull items that are (instance of OR subclass of) music genre.
    """
    # This query asks: give me items that are music genres (or subclasses),
    # plus their English labels/aliases, and their English Wikipedia sitelink.
    # Q188451 = music genre.
    query = """
    SELECT ?item ?itemLabel ?enwiki ?enwikiTitle ?aliases WHERE {
      {
        ?item wdt:P31/wdt:P279* wd:Q188451 .
      } UNION {
        ?item wdt:P279* wd:Q188451 .
      }
      OPTIONAL {
        ?enwiki schema:about ?item ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?enwikiTitle .
        BIND(STR(?enwiki) AS ?enwiki)
      }
      OPTIONAL {
        ?item skos:altLabel ?aliases .
        FILTER(LANG(?aliases) = "en")
      }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    rows = wdqs_query(query)

    # Build mapping: label/alias -> page
    out: Dict[str, Tuple[str, str, List[str]]] = {}

    # Collect per item first (so aliases join properly)
    by_item: Dict[str, Dict] = {}
    for r in rows:
        item = r.get("item")
        if not item:
            continue
        d = by_item.setdefault(item, {"label": None, "title": None, "url": None, "aliases": set()})
        if r.get("itemLabel"):
            d["label"] = r["itemLabel"]
        if r.get("enwikiTitle"):
            d["title"] = r["enwikiTitle"]
        if r.get("enwiki"):
            d["url"] = r["enwiki"]
        if r.get("aliases"):
            d["aliases"].add(r["aliases"])

    for item, d in by_item.items():
        title = d.get("title")
        url = d.get("url")
        label = d.get("label")
        aliases = sorted(list(d.get("aliases") or []))

        if not title or not url or not label:
            continue

        # Index label and aliases
        keys = [label] + aliases
        for k in keys:
            nk = norm(k)
            if not nk:
                continue
            # Keep first seen (usually fine). Overrides will win later anyway.
            if nk not in out:
                out[nk] = (title, url, aliases)

    return out

def wikipedia_intro_looks_like_music(intro: str) -> bool:
    """Very conservative check to reject cities/people pages."""
    if not intro:
        return False
    p = intro.lower()
    music_terms = ["music", "musical", "genre", "style", "band", "song", "album"]
    place_terms = ["city", "town", "county", "province", "state", "capital", "population", "located in"]
    if not any(t in p for t in music_terms):
        return False
    # reject strong place intros unless also clearly music-related
    if any(t in p for t in place_terms) and ("music" not in p and "musical" not in p):
        return False
    return True

def wikipedia_best_title_fallback(genre: str) -> Optional[str]:
    """
    Fallback when Wikidata doesn't have it:
    search Wikipedia, but only accept pages whose intro looks music-related.
    """
    headers = {"User-Agent": UA}
    search_queries = [
        f'intitle:"{genre}" (music OR genre OR style)',
        f'"{genre}" (music OR genre OR style)',
        f"{genre} music genre",
        f"{genre} musical style",
    ]

    for q in search_queries:
        try:
            s = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": q, "srlimit": 5, "format": "json"},
                headers=headers,
                timeout=20,
            )
            s.raise_for_status()
            hits = s.json().get("query", {}).get("search", [])
            for h in hits:
                title = h.get("title")
                if not title:
                    continue

                # fetch intro using TextExtracts (fast)
                e = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "extracts|info",
                        "exintro": 1,
                        "explaintext": 1,
                        "inprop": "url",
                        "redirects": 1,
                        "titles": title,
                        "format": "json",
                    },
                    headers=headers,
                    timeout=20,
                )
                e.raise_for_status()
                pages = e.json().get("query", {}).get("pages", {})
                page = next(iter(pages.values())) if pages else {}
                intro = (page.get("extract") or "").strip().split("\n")[0].strip()
                if wikipedia_intro_looks_like_music(intro):
                    return page.get("title") or title

            time.sleep(0.2)  # gentle
        except Exception:
            continue

    return None

def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    # Input: your genre list (downloaded already by the app, but we use the same CSV)
    input_csv = data_dir / "genre_attrs.csv"
    if not input_csv.exists():
        raise SystemExit(
            "Missing data/genre_attrs.csv. Put a copy of your genre CSV there first.\n"
            "Tip: download it from the app’s DATA_URL and save it into data/genre_attrs.csv."
        )

    overrides_path = data_dir / "wiki_overrides.csv"
    out_path = data_dir / "wiki_map.csv"
    needs_review_path = data_dir / "wiki_needs_review.csv"

    overrides = load_overrides(overrides_path)

    print("Downloading Wikidata music-genre index (this can take a bit)...")
    wd_index = fetch_wikidata_music_genres()
    print(f"Indexed {len(wd_index):,} label/alias strings from Wikidata.")

    df = pd.read_csv(input_csv)
    if "genre" not in df.columns:
        raise SystemExit("Input CSV must contain a 'genre' column.")

    rows_out = []
    rows_review = []

    for g in df["genre"].astype(str).tolist():
        ng = norm(g)

        # 1) manual override wins
        if ng in overrides:
            rows_out.append({"genre": g, "wiki_title": overrides[ng], "confidence": "high", "source": "override"})
            continue

        # 2) Wikidata exact/alias
        if ng in wd_index:
            title, _url, _aliases = wd_index[ng]
            rows_out.append({"genre": g, "wiki_title": title, "confidence": "high", "source": "wikidata"})
            continue

        # 3) Fallback: Wikipedia search with strict music-only checks
        title = wikipedia_best_title_fallback(g)
        if title:
            rows_out.append({"genre": g, "wiki_title": title, "confidence": "medium", "source": "wikipedia_fallback"})
        else:
            rows_out.append({"genre": g, "wiki_title": "", "confidence": "none", "source": "unmatched"})
            rows_review.append({"genre": g})

    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    pd.DataFrame(rows_review).to_csv(needs_review_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Needs review: {needs_review_path} (unmatched genres)")

if __name__ == "__main__":
    main()
