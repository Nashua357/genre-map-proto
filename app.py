import base64
import heapq
import html
import re
import time
import zlib
from typing import Optional, List, Tuple, Dict, Set
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.neighbors import NearestNeighbors

# =========================
# URLs
# =========================
DATA_URL = "https://raw.githubusercontent.com/AyrtonB/EveryNoise-Watch/main/data/genre_attrs.csv"

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_UA = "genre-map-proto/1.0 (personal project)"

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
NONE_OPTION = "— none —"

# background color used for blending (matches plotly_dark feel)
BG_HEX = "#0b0f17"


# =========================
# Helpers
# =========================
def safe_hex(s: str) -> str:
    if s is None:
        return "#888888"
    s = str(s).strip()
    if not s:
        return "#888888"
    if not s.startswith("#"):
        s = "#" + s
    if HEX_RE.match(s):
        return s
    return "#888888"


def norm_title(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_tags(s: str) -> str:
    s = re.sub(r"<.*?>", "", s or "")
    return html.unescape(s).strip()


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = safe_hex(h).lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def blend_hex(fg_hex: str, bg_hex: str, fg_strength: float) -> str:
    """
    fg_strength = 1.0 => fg
    fg_strength = 0.0 => bg
    """
    fg_strength = float(np.clip(fg_strength, 0.0, 1.0))
    fr, fg, fb = hex_to_rgb(fg_hex)
    br, bg, bb = hex_to_rgb(bg_hex)
    r = int(round(br + (fr - br) * fg_strength))
    g = int(round(bg + (fg - bg) * fg_strength))
    b = int(round(bb + (fb - bb) * fg_strength))
    return rgb_to_hex((r, g, b))


# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False)
def load_genre_data(source: str, uploaded_bytes: Optional[bytes] = None) -> pd.DataFrame:
    if source == "web":
        df = pd.read_csv(DATA_URL)
    else:
        if uploaded_bytes is None:
            raise ValueError("No uploaded file provided.")
        df = pd.read_csv(uploaded_bytes)

    required = {"genre", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=["genre", "x", "y"]).copy()
    df["genre"] = df["genre"].astype(str)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).copy()

    # Colors: accept r/g/b or hex_colour
    if {"r", "g", "b"}.issubset(df.columns):
        for c in ["r", "g", "b"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["r", "g", "b"]).copy()
        df["r"] = df["r"].clip(0, 255).astype(int)
        df["g"] = df["g"].clip(0, 255).astype(int)
        df["b"] = df["b"].clip(0, 255).astype(int)
        df["hex_colour"] = (
            "#" + df["r"].map(lambda v: f"{v:02x}")
            + df["g"].map(lambda v: f"{v:02x}")
            + df["b"].map(lambda v: f"{v:02x}")
        )
    else:
        hex_col = None
        for c in ["hex_colour", "hex_color", "hex", "color", "colour"]:
            if c in df.columns:
                hex_col = c
                break
        if hex_col is None:
            df["hex_colour"] = "#888888"
        else:
            df["hex_colour"] = df[hex_col]

    df["hex_colour"] = df["hex_colour"].apply(safe_hex)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def stable_sample_genres(genres: pd.Series, n: int = 500) -> List[str]:
    hashes = genres.astype(str).apply(lambda s: zlib.crc32(s.encode("utf-8")))
    picked_idx = hashes.nsmallest(min(n, len(genres))).index
    return genres.loc[picked_idx].sort_values().tolist()


@st.cache_data(show_spinner=False)
def build_knn_graph(coords: np.ndarray, k: int):
    n = coords.shape[0]
    k = max(1, min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    edges: List[Tuple[int, int, float]] = []

    for u in range(n):
        for pos in range(1, k + 1):
            v = int(indices[u, pos])
            d = float(distances[u, pos])
            adj[u].append((v, d))
            adj[v].append((u, d))
            if u < v:
                edges.append((u, v, d))
    return adj, edges


def dijkstra_path(adj: List[List[Tuple[int, float]]], start: int, goal: int) -> List[int]:
    if start == goal:
        return [start]
    INF = 1e18
    dist = {start: 0.0}
    prev: Dict[int, int] = {}
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        if d != dist.get(u, INF):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in dist:
        return []

    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# =========================
# Wikipedia (lead + links)
# =========================
def _fetch_page_intro_by_title(title: str) -> Dict[str, Optional[str]]:
    headers = {"User-Agent": WIKI_UA}
    r = requests.get(
        WIKI_API,
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
        timeout=15,
    )
    r.raise_for_status()
    js = r.json()
    pages = js.get("query", {}).get("pages", {})
    page = next(iter(pages.values())) if pages else {}
    if page.get("missing") is not None:
        return {"title": None, "paragraph": None, "url": None}

    extract = (page.get("extract") or "").strip()
    paragraph = extract.split("\n")[0].strip() if extract else None
    url = page.get("fullurl")
    return {"title": page.get("title"), "paragraph": paragraph, "url": url}


def _fetch_lead_html(title: str) -> Optional[str]:
    headers = {"User-Agent": WIKI_UA}
    r = requests.get(
        WIKI_API,
        params={
            "action": "parse",
            "page": title,
            "prop": "text",
            "section": 0,
            "format": "json",
            "redirects": 1,
        },
        headers=headers,
        timeout=15,
    )
    r.raise_for_status()
    js = r.json()
    return js.get("parse", {}).get("text", {}).get("*")


def _extract_first_paragraph_links(lead_html: str) -> List[str]:
    if not lead_html:
        return []
    ps = re.findall(r"<p>(.*?)</p>", lead_html, flags=re.DOTALL | re.IGNORECASE)
    first_p = None
    for p in ps:
        plain = strip_tags(p)
        if len(plain) > 60:
            first_p = p
            break
    if not first_p:
        return []

    anchors = re.findall(r'<a[^>]*title="([^"]+)"[^>]*>(.*?)</a>', first_p, flags=re.DOTALL | re.IGNORECASE)
    names = []
    for title_attr, inner in anchors:
        t = strip_tags(inner)
        ta = strip_tags(title_attr)
        if not t:
            continue
        if ":" in ta or ta.lower().startswith("list of "):
            continue
        names.append(t)

    out = []
    seen = set()
    for n in names:
        k = n.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_wikipedia_lead(genre_name: str) -> Dict[str, object]:
    headers = {"User-Agent": WIKI_UA}
    g = (genre_name or "").strip()
    if not g:
        return {"title": None, "paragraph": None, "url": None, "lead_link_names": []}

    banned_titles = {
        "music genre",
        "music",
        "genre",
        "popular music",
        "list of genres",
        "outline of music",
    }

    best = {"title": None, "paragraph": None, "url": None}

    # exact title first
    try:
        direct = _fetch_page_intro_by_title(g)
        if direct.get("title") and norm_title(direct["title"]) not in banned_titles and direct.get("paragraph"):
            best = direct
    except Exception:
        pass

    # search candidates
    if not best.get("title"):
        candidates: List[str] = []
        queries = [
            f'intitle:"{g}"',
            f'"{g}"',
            f"{g} music",
            f"{g} genre",
        ]
        for q in queries:
            try:
                r = requests.get(
                    WIKI_API,
                    params={"action": "query", "list": "search", "srsearch": q, "srlimit": 5, "format": "json"},
                    headers=headers,
                    timeout=15,
                )
                r.raise_for_status()
                hits = r.json().get("query", {}).get("search", [])
                for h in hits:
                    t = h.get("title")
                    if t:
                        candidates.append(t)
            except Exception:
                continue

        seen = set()
        cand_titles = []
        for t in candidates:
            if t not in seen:
                seen.add(t)
                cand_titles.append(t)

        ng = norm_title(g)

        def score_title(t: str) -> int:
            nt = norm_title(t)
            if nt in banned_titles:
                return -9999
            if nt == ng:
                return 100
            if ng and ng in nt:
                return 80
            toks = ng.split()
            if toks and all(tok in nt for tok in toks):
                return 60
            return 10

        best_score = -9999
        for t in cand_titles[:12]:
            s = score_title(t)
            if s <= best_score:
                continue
            try:
                page = _fetch_page_intro_by_title(t)
                if not page.get("paragraph"):
                    continue
                if norm_title(page.get("title") or "") in banned_titles:
                    continue
                best = page
                best_score = s
            except Exception:
                continue

    lead_links: List[str] = []
    if best.get("title"):
        try:
            lead_html = _fetch_lead_html(best["title"])
            lead_links = _extract_first_paragraph_links(lead_html or "")
        except Exception:
            lead_links = []

    return {
        "title": best.get("title"),
        "paragraph": best.get("paragraph"),
        "url": best.get("url"),
        "lead_link_names": lead_links,
    }


# =========================
# Spotify API
# =========================
def _get_spotify_creds() -> Tuple[Optional[str], Optional[str]]:
    try:
        return st.secrets.get("SPOTIFY_CLIENT_ID"), st.secrets.get("SPOTIFY_CLIENT_SECRET")
    except Exception:
        return None, None


def get_spotify_token() -> Optional[str]:
    now = time.time()
    tok = st.session_state.get("spotify_token")
    exp = st.session_state.get("spotify_token_exp", 0.0)
    if tok and now < exp - 60:
        return tok

    client_id, client_secret = _get_spotify_creds()
    if not client_id or not client_secret:
        return None

    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    try:
        r = requests.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {auth}"},
            timeout=15,
        )
        r.raise_for_status()
        js = r.json()
        tok = js.get("access_token")
        expires_in = float(js.get("expires_in", 3600))
        if not tok:
            return None
        st.session_state["spotify_token"] = tok
        st.session_state["spotify_token_exp"] = time.time() + expires_in
        return tok
    except Exception:
        return None


def spotify_search_artist(name: str, market: str = "AU") -> Optional[Dict[str, str]]:
    tok = get_spotify_token()
    if not tok:
        return None
    headers = {"Authorization": f"Bearer {tok}"}

    try:
        r = requests.get(
            SPOTIFY_SEARCH_URL,
            params={"q": f'artist:"{name}"', "type": "artist", "limit": 1, "market": market},
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("artists", {}).get("items", [])
        if not items:
            return None
        a = items[0]
        return {
            "type": "artist",
            "id": a["id"],
            "name": a.get("name", name),
            "url": a.get("external_urls", {}).get("spotify", ""),
        }
    except Exception:
        return None


def spotify_artist_from_wikipedia_links(link_names: List[str], market: str = "AU") -> Optional[Dict[str, str]]:
    if not link_names:
        return None

    blacklist = {
        "music", "genre", "rock", "jazz", "pop", "hip hop", "hip-hop", "blues",
        "united states", "england", "australia", "british", "american",
        "song", "album", "record", "band", "singer", "musician",
    }

    def candidate_score(n: str) -> int:
        nn = norm_title(n)
        if not nn:
            return -9999
        if nn in blacklist:
            return -9999
        if len(nn) < 3 or len(nn) > 45:
            return -9999
        words = nn.split()
        if 2 <= len(words) <= 4:
            return 100
        if len(words) == 1:
            return 55
        return 25

    ranked = sorted(link_names, key=candidate_score, reverse=True)
    ranked = [n for n in ranked if candidate_score(n) > 0][:10]

    for name in ranked:
        a = spotify_search_artist(name, market=market)
        if not a:
            continue
        if norm_title(a["name"]) == norm_title(name) or norm_title(name) in norm_title(a["name"]):
            return a

    for name in ranked:
        a = spotify_search_artist(name, market=market)
        if a:
            return a

    return None


def spotify_example_for_genre(genre_name: str, market: str = "AU") -> Optional[Dict[str, str]]:
    cache = st.session_state.setdefault("spotify_example_cache", {})
    key = ("genre_example", genre_name, market)
    if key in cache:
        return cache[key]

    tok = get_spotify_token()
    if not tok:
        return None
    headers = {"Authorization": f"Bearer {tok}"}

    try:
        r = requests.get(
            SPOTIFY_SEARCH_URL,
            params={"q": f'genre:"{genre_name}"', "type": "track", "limit": 1, "market": market},
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("tracks", {}).get("items", [])
        if items:
            t = items[0]
            result = {
                "type": "track",
                "id": t["id"],
                "name": t["name"],
                "subtitle": t["artists"][0]["name"] if t.get("artists") else "",
                "url": t.get("external_urls", {}).get("spotify", ""),
            }
            cache[key] = result
            return result
    except Exception:
        pass

    try:
        r = requests.get(
            SPOTIFY_SEARCH_URL,
            params={"q": genre_name, "type": "playlist", "limit": 1, "market": market},
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("playlists", {}).get("items", [])
        if items:
            p = items[0]
            result = {
                "type": "playlist",
                "id": p["id"],
                "name": p["name"],
                "subtitle": "Playlist",
                "url": p.get("external_urls", {}).get("spotify", ""),
            }
            cache[key] = result
            return result
    except Exception:
        pass

    return None


def spotify_embed_html(item_type: str, item_id: str, height: int = 152) -> str:
    if item_type not in {"track", "playlist", "artist"}:
        item_type = "track"
    src = f"https://open.spotify.com/embed/{item_type}/{item_id}"
    return f"""
      <iframe style="border-radius:12px"
        src="{src}"
        width="100%" height="{height}" frameborder="0"
        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
        loading="lazy"></iframe>
    """


# =========================
# Selection history
# =========================
def push_history(genre: str, max_len: int = 20):
    if not genre or genre == NONE_OPTION:
        return
    hist = st.session_state.setdefault("genre_history", [])
    if len(hist) == 0 or hist[0] != genre:
        hist = [genre] + [g for g in hist if g != genre]
        st.session_state["genre_history"] = hist[:max_len]


# =========================
# Figure (stable traces -> no reset)
# =========================
def build_edges_xy(X: np.ndarray, Y: np.ndarray, edges: List[Tuple[int, int, float]]) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for u, v, _d in edges:
        xs.extend([float(X[u]), float(X[v]), None])
        ys.extend([float(Y[u]), float(Y[v]), None])
    return xs, ys


def build_star_xy(X: np.ndarray, Y: np.ndarray, center_idx: int, adj: List[List[Tuple[int, float]]]) -> Tuple[List[float], List[float]]:
    sx, sy = float(X[center_idx]), float(Y[center_idx])
    xs, ys = [], []
    for v, _d in adj[center_idx]:
        xs.extend([sx, float(X[v]), None])
        ys.extend([sy, float(Y[v]), None])
    return xs, ys


def build_path_xy(X: np.ndarray, Y: np.ndarray, path: List[int]) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for a, b in zip(path[:-1], path[1:]):
        xs.extend([float(X[a]), float(X[b]), None])
        ys.extend([float(Y[a]), float(Y[b]), None])
    return xs, ys


def make_figure(
    df_plot: pd.DataFrame,
    selected_idx: Optional[int],
    neighbor_idxs: Set[int],
    edges_all: List[Tuple[int, int, float]],
    adj: List[List[Tuple[int, float]]],
    show_edges: bool,
    path: List[int],
    show_labels: bool,
    neighbor_label_count: int,
) -> go.Figure:
    n = len(df_plot)
    X = df_plot["x"].astype(float).to_numpy()
    Y = df_plot["y"].astype(float).to_numpy()
    C = df_plot["hex_colour"].to_numpy()
    CF_SEL = df_plot["hex_faint_sel"].to_numpy()
    CF_PATH = df_plot["hex_faint_path"].to_numpy()
    G = df_plot["genre"].to_numpy()
    IDX = np.arange(n)

    has_selection = selected_idx is not None
    has_path = len(path) >= 2

    # stable base edges
    edges_x, edges_y = build_edges_xy(X, Y, edges_all)

    # choose base color set
    if not has_selection:
        base_colors = C
    else:
        base_colors = CF_PATH if has_path else CF_SEL

    # highlight nodes
    hi_idx: List[int] = []
    if has_selection and not has_path:
        hi_idx = sorted(set(list(neighbor_idxs) + [selected_idx]))
    elif has_selection and has_path:
        hi_idx = sorted(set([i for i in path if 0 <= i < n]))

    # highlight lines (star or path)
    hi_lx, hi_ly = [], []
    if has_selection and show_edges:
        if has_path:
            hi_lx, hi_ly = build_path_xy(X, Y, [i for i in path if 0 <= i < n])
        else:
            hi_lx, hi_ly = build_star_xy(X, Y, selected_idx, adj)

    # labels (limited)
    label_x, label_y, label_t = [], [], []
    if has_selection and show_labels:
        label_idxs: List[int] = []
        if has_path:
            if 0 <= path[0] < n:
                label_idxs.append(path[0])
            if 0 <= path[-1] < n:
                label_idxs.append(path[-1])
        else:
            label_idxs.append(selected_idx)
            if neighbor_label_count > 0:
                for i in sorted(list(neighbor_idxs))[:neighbor_label_count]:
                    label_idxs.append(i)
        label_idxs = sorted(set([i for i in label_idxs if 0 <= i < n]))
        label_x = [float(X[i]) for i in label_idxs]
        label_y = [float(Y[i]) for i in label_idxs]
        label_t = [str(G[i]) for i in label_idxs]

    # make figure with ALWAYS the same trace structure
    fig = go.Figure()

    # 0) global edges (always present; just fade when selected)
    fig.add_trace(
        go.Scattergl(
            x=edges_x,
            y=edges_y,
            mode="lines",
            line=dict(width=1.1, color="rgba(150,180,255,0.34)" if not has_selection else "rgba(150,180,255,0.10)"),
            hoverinfo="skip",
            showlegend=False,
            uid="edges_all",
            visible=True if show_edges else False,
        )
    )

    # 1) base points (always present)
    fig.add_trace(
        go.Scattergl(
            x=X,
            y=Y,
            mode="markers",
            marker=dict(size=9, color=base_colors, opacity=1.0),
            customdata=IDX,
            hoverinfo="skip",
            showlegend=False,
            uid="points_base",
        )
    )

    # 2) big invisible click targets (always present)
    fig.add_trace(
        go.Scattergl(
            x=X,
            y=Y,
            mode="markers",
            marker=dict(size=24, color="rgba(0,0,0,0.001)", opacity=0.001),
            customdata=IDX,
            hoverinfo="skip",
            showlegend=False,
            uid="hit_targets",
        )
    )

    # 3) highlight line(s) (always present; sometimes empty)
    fig.add_trace(
        go.Scattergl(
            x=hi_lx,
            y=hi_ly,
            mode="lines",
            line=dict(width=4.0, color="rgba(120,200,255,0.92)"),
            hoverinfo="skip",
            showlegend=False,
            uid="lines_highlight",
        )
    )

    # 4) highlight points (always present; sometimes empty)
    if hi_idx:
        fig.add_trace(
            go.Scattergl(
                x=X[hi_idx],
                y=Y[hi_idx],
                mode="markers",
                marker=dict(size=12, color=C[hi_idx], opacity=1.0),
                customdata=IDX[hi_idx],
                hoverinfo="skip",
                showlegend=False,
                uid="points_highlight",
            )
        )
    else:
        fig.add_trace(
            go.Scattergl(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=12, color=[]),
                hoverinfo="skip",
                showlegend=False,
                uid="points_highlight",
            )
