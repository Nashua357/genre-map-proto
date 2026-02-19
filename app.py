"""
Genre Map Prototype — Phase 1
Streamlit app: interactive EveryNoise-derived genre map with
Wikipedia summaries and Spotify example tracks.

Requirements (requirements.txt):
    streamlit
    pandas
    numpy
    scikit-learn
    plotly
    requests

Repo structure expected:
    app.py
    data/
        genre_attrs.csv        (genre, x, y, r, g, b  OR  genre, x, y, hex_colour)
        wiki_map.csv           (genre, wiki_title)        ← optional but recommended
        wiki_overrides.csv     (genre, wiki_title)        ← optional manual overrides

Streamlit Secrets required (Settings → Secrets in Streamlit Cloud):
    SPOTIFY_CLIENT_ID     = "..."
    SPOTIFY_CLIENT_SECRET = "..."
"""

import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config + global style
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Genre Map Prototype", layout="wide")
st.markdown(
    """
    <style>
    .small-muted { color: rgba(255,255,255,0.55); font-size: 0.9rem; }
    .stSelectbox label { cursor: pointer; }
    .stSelectbox div[data-baseweb="select"] { cursor: pointer; }
    .stTextInput label { cursor: text; }
    iframe[src*="spotify"] { border: none; }
    .stHtml { overflow: hidden !important; }
    div[data-testid="stHtml"] { overflow: hidden !important; }
    div[data-testid="stHtml"] > div { overflow: hidden !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants / endpoints
# ---------------------------------------------------------------------------
WIKI_API          = "https://en.wikipedia.org/w/api.php"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE  = "https://api.spotify.com/v1"
PLOT_KEY          = "genre_map_plot"

# ---------------------------------------------------------------------------
# Session state initialisation  (MUST be before any widgets)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "selected_genre":     "(none)",   # widget-bound (selectbox)
    "destination_genre":  "(none)",   # widget-bound (selectbox)
    "_active_genre":      "",         # NOT widget-bound — true current genre
    "selection_history":  [],
    "active_path":        [],
    "market":             "AU",
    "reset_view_tick":    0,
    "_pending_click":     None,       # bridge: fragment click → selectbox
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Process any pending click BEFORE selectbox renders so the dropdown shows
# the clicked genre.  _pending_click is set by the fragment on dot-click
# and consumed here on the next FULL rerun (triggered by any non-fragment
# widget interaction).
if st.session_state._pending_click is not None:
    st.session_state.selected_genre  = st.session_state._pending_click
    st.session_state._active_genre   = st.session_state._pending_click
    st.session_state.active_path     = []
    st.session_state._pending_click  = None

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def norm_title(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_data(show_spinner=False)
def http_get(url: str, params: dict, timeout: int = 30) -> dict:
    headers = {
        "User-Agent": "genre-map-proto/1.0 (personal project)",
        "Accept": "application/json",
    }
    for attempt in range(6):
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            sleep_s = int(retry) if retry and retry.isdigit() else (2 + attempt * 3)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Too many rate-limit retries for {url}")


# ---------------------------------------------------------------------------
# Spotify helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3300)
def spotify_get_token() -> Optional[str]:
    client_id     = st.secrets.get("SPOTIFY_CLIENT_ID", None)
    client_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET", None)
    if not client_id or not client_secret:
        return None
    try:
        r = requests.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception:
        return None


def spotify_api_get(path: str, token: str, params: dict, timeout: int = 30) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(SPOTIFY_API_BASE + path, headers=headers, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def spotify_search_track(token: str, query: str, market: str) -> Optional[dict]:
    js    = spotify_api_get("/search", token, params={"q": query, "type": "track", "limit": 5, "market": market})
    items = js.get("tracks", {}).get("items", [])
    return items[0] if items else None


def spotify_embed_html(track_id: str) -> str:
    return (
        f'<iframe style="border-radius:12px; border:none;" '
        f'src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator" '
        f'width="100%" height="152" frameborder="0" scrolling="no" '
        f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; '
        f'picture-in-picture" loading="lazy"></iframe>'
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _parse_df(df: pd.DataFrame) -> pd.DataFrame:
    colmap     = {c.lower(): c for c in df.columns}
    genre_col  = colmap.get("genre") or colmap.get("name")
    x_col      = colmap.get("x")
    y_col      = colmap.get("y")
    if not genre_col or not x_col or not y_col:
        raise ValueError("CSV must contain columns: genre (or name), x, y")
    df = df.rename(columns={genre_col: "genre", x_col: "x", y_col: "y"})

    r_col, g_col, b_col = colmap.get("r"), colmap.get("g"), colmap.get("b")
    if r_col and g_col and b_col:
        df = df.rename(columns={r_col: "r", g_col: "g", b_col: "b"})
    else:
        hex_col = None
        for c in ["hex_colour", "hex_color", "hex", "color", "colour"]:
            if c in colmap:
                hex_col = colmap[c]
                break
        if hex_col:
            hh = df[hex_col].astype(str).str.strip().str.lstrip("#")
            df["r"] = hh.str[0:2].apply(lambda s: int(s, 16) if len(s) >= 2 else 136)
            df["g"] = hh.str[2:4].apply(lambda s: int(s, 16) if len(s) >= 2 else 136)
            df["b"] = hh.str[4:6].apply(lambda s: int(s, 16) if len(s) >= 2 else 136)
        else:
            df["r"], df["g"], df["b"] = 180, 180, 180

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["r"] = pd.to_numeric(df["r"], errors="coerce").fillna(180).clip(0, 255).astype(int)
    df["g"] = pd.to_numeric(df["g"], errors="coerce").fillna(180).clip(0, 255).astype(int)
    df["b"] = pd.to_numeric(df["b"], errors="coerce").fillna(180).clip(0, 255).astype(int)
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    df["genre"] = df["genre"].astype(str)
    df["hex"]   = df.apply(lambda row: f"rgb({int(row.r)},{int(row.g)},{int(row.b)})", axis=1)
    return df


@st.cache_data(show_spinner=False)
def load_genre_data_from_repo() -> pd.DataFrame:
    base = Path(__file__).resolve().parent
    path = base / "data" / "genre_attrs.csv"
    if not path.exists():
        raise FileNotFoundError("Missing data/genre_attrs.csv in the repo.")
    return _parse_df(pd.read_csv(path))


def load_genre_data_from_upload(uploaded_file) -> pd.DataFrame:
    return _parse_df(pd.read_csv(uploaded_file))


# ---------------------------------------------------------------------------
# Wikipedia mapping helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_wiki_title_maps() -> Tuple[dict, dict]:
    base           = Path(__file__).resolve().parent
    overrides_path = base / "data" / "wiki_overrides.csv"
    map_path       = base / "data" / "wiki_map.csv"
    overrides: dict = {}
    generated: dict = {}

    if overrides_path.exists():
        odf = pd.read_csv(overrides_path)
        if "genre" in odf.columns and "wiki_title" in odf.columns:
            for _, row in odf.iterrows():
                g = norm_title(str(row.get("genre", "")))
                t = str(row.get("wiki_title", "")).strip()
                if g and t:
                    overrides[g] = t

    if map_path.exists():
        mdf = pd.read_csv(map_path)
        if "genre" in mdf.columns and "wiki_title" in mdf.columns:
            for _, row in mdf.iterrows():
                g = norm_title(str(row.get("genre", "")))
                t = str(row.get("wiki_title", "")).strip()
                if g and t and str(t).lower() != "nan":
                    generated[g] = t

    return overrides, generated


def resolve_wiki_title_for_genre(genre: str) -> Optional[str]:
    overrides_map, generated_map = load_wiki_title_maps()
    key = norm_title(genre)
    return overrides_map.get(key) or generated_map.get(key)


@st.cache_data(show_spinner=False)
def wiki_intro_by_title(title: str) -> dict:
    js = http_get(
        WIKI_API,
        params={
            "action": "query", "prop": "extracts", "exintro": 1,
            "explaintext": 1, "redirects": 1, "titles": title, "format": "json",
        },
    )
    pages     = js.get("query", {}).get("pages", {})
    page      = next(iter(pages.values())) if pages else {}
    t         = page.get("title") or title
    extract   = (page.get("extract") or "").strip()
    paragraph = extract.split("\n")[0].strip() if extract else ""
    url       = f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}" if t else ""
    return {"title": t, "paragraph": paragraph, "url": url}


@st.cache_data(show_spinner=False)
def wiki_lead_html(title: str) -> str:
    js = http_get(
        WIKI_API,
        params={"action": "parse", "page": title, "prop": "text", "section": 0, "format": "json"},
    )
    return js.get("parse", {}).get("text", {}).get("*") or ""


def extract_first_paragraph_link_titles(html: str, max_items: int = 25) -> List[str]:
    if not html:
        return []
    m = re.search(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    p      = m.group(1)
    titles: List[str] = []
    for tm in re.finditer(r'title="([^"]+)"', p):
        t = tm.group(1)
        if t and t not in titles:
            titles.append(t)
        if len(titles) >= max_items:
            break
    if not titles:
        for wm in re.finditer(r'href="/wiki/([^"#?]+)"', p):
            t = wm.group(1).replace("_", " ")
            if t and t not in titles:
                titles.append(t)
            if len(titles) >= max_items:
                break
    return titles[:max_items]


def pick_spotify_example_from_wikipedia(
    token: Optional[str], wiki_title: str, market: str
) -> Optional[dict]:
    if not token:
        return None
    try:
        html         = wiki_lead_html(wiki_title)
        link_titles  = extract_first_paragraph_link_titles(html, max_items=25)
    except Exception:
        link_titles  = []

    filtered_titles: List[str] = []
    for t in link_titles:
        tl = t.lower()
        if any(x in tl for x in ["list of", "category:", "wikipedia:", "help:", "file:"]):
            continue
        if any(x in tl for x in ["music", "band", "singer", "rapper", "producer", "artist", "musician"]):
            filtered_titles.append(t)
            continue
        if len(t) <= 40 and not any(x in tl for x in ["city", "county", "province", "country"]):
            filtered_titles.append(t)

    for t in filtered_titles[:12]:
        q = f'{t} "{wiki_title}"'
        try:
            track = spotify_search_track(token, q, market=market)
            if track:
                track["_picked_from_artist"] = t
                return track
        except Exception:
            continue

    try:
        return spotify_search_track(token, f"{wiki_title} genre", market=market)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Neighbour graph
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_neighbors(xy: np.ndarray, k: int = 8) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(xy)), algorithm="auto")
    nn.fit(xy)
    _dists, idxs = nn.kneighbors(xy)
    return idxs[:, 1:]


def build_adjacency(neighbors: np.ndarray) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(neighbors.shape[0])]
    for i in range(neighbors.shape[0]):
        for j in neighbors[i]:
            adj[i].append(int(j))
            adj[int(j)].append(i)
    return [sorted(set(nbrs)) for nbrs in adj]


def shortest_path(adj: List[List[int]], start: int, goal: int) -> List[int]:
    if start == goal:
        return [start]
    from collections import deque
    q    = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in prev:
                prev[nxt] = cur
                if nxt == goal:
                    q.clear()
                    break
                q.append(nxt)
    if goal not in prev:
        return []
    path = []
    node: Optional[int] = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    return list(reversed(path))


def pick_label_points(df: pd.DataFrame, max_labels: int = 120) -> np.ndarray:
    max_labels   = int(clamp(max_labels, 10, 500))
    x            = df["x"].to_numpy()
    y            = df["y"].to_numpy()
    bins         = int(clamp(int(np.sqrt(max_labels) * 1.5), 8, 45))
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    gx = np.floor((x - x_min) / (x_max - x_min + 1e-9) * bins).astype(int)
    gy = np.floor((y - y_min) / (y_max - y_min + 1e-9) * bins).astype(int)
    chosen: dict = {}
    for idx in range(len(df)):
        key = (gx[idx], gy[idx])
        if key not in chosen:
            chosen[key] = idx
        if len(chosen) >= max_labels:
            break
    return np.array(list(chosen.values()), dtype=int)


# ---------------------------------------------------------------------------
# History helper
# ---------------------------------------------------------------------------
def push_history(genre: str, max_items: int = 12) -> None:
    g = (genre or "").strip()
    if not g or g == "(none)":
        return
    hist = list(st.session_state.selection_history)
    if g in hist:
        hist.remove(g)
    hist.insert(0, g)
    st.session_state.selection_history = hist[:max_items]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Selection history")
    st.caption("Click dots or pick genres to build history.")
    if st.session_state.selection_history:
        for g in st.session_state.selection_history:
            if st.button(g, use_container_width=True, key=f"hist_{g}"):
                st.session_state._pending_click = g
                st.rerun()
        cols = st.columns([1, 1])
        if cols[0].button("Clear", use_container_width=True):
            st.session_state.selection_history = []
            st.rerun()
        cols[1].markdown(
            f"<div class='small-muted'>{len(st.session_state.selection_history)} genres</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='small-muted'>No history yet.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Data")
    st.caption("Genres come from your EveryNoise-derived CSV: data/genre_attrs.csv")
    data_source = st.radio("Load dataset from:", ["Repo CSV (recommended)", "Upload CSV"])
    uploaded    = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    st.markdown("## Connections")
    k          = st.slider("Connections per genre (nearest neighbors)", min_value=2, max_value=20, value=8)
    show_lines = st.checkbox("Show connection lines", value=True)
    enable_path = st.checkbox("Enable path finder", value=True)

    st.markdown("---")
    st.markdown("## Labels")
    show_labels        = st.checkbox("Show labels on map", value=True)
    default_label_count = st.slider("Default label count", 30, 220, 80, 10)
    label_size         = st.slider("Label size", 8, 18, 11, 1)

    st.markdown("---")
    st.markdown("## Visibility")
    fade_others   = st.slider("Fade non-selected dots", 0, 90, 65, 5)
    line_strength = st.slider("Line strength", 0, 100, 60, 5)
    dot_size      = st.slider("Dot size", 4, 14, 8, 1)

    st.markdown("---")
    st.markdown("## Spotify examples")
    st.session_state.market = st.selectbox(
        "Country/market", ["AU", "US", "GB", "CA", "NZ", "DE", "FR", "JP", "BR"], index=0
    )

    st.markdown("---")
    st.markdown("## View")
    map_fit = st.radio("Map shape", ["Fit to screen", "Original"], index=0)
    if st.button("Reset map view", use_container_width=True):
        st.session_state.reset_view_tick += 1
        st.rerun()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    if data_source == "Upload CSV" and uploaded is not None:
        df = load_genre_data_from_upload(uploaded)
    else:
        df = load_genre_data_from_repo()
    if df is None or df.empty:
        raise ValueError("No data loaded.")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

genres       = df["genre"].tolist()
genre_to_idx = {g: i for i, g in enumerate(genres)}
xy           = df[["x", "y"]].to_numpy(dtype=np.float32)
neighbors    = compute_neighbors(xy, k=k)
adj          = build_adjacency(neighbors)
label_idxs_default = pick_label_points(df, max_labels=default_label_count)

x_min, x_max = float(df["x"].min()), float(df["x"].max())
y_min, y_max = float(df["y"].min()), float(df["y"].max())
pad_x        = (x_max - x_min) * 0.04
pad_y        = (y_max - y_min) * 0.04
default_xrange = [x_min - pad_x, x_max + pad_x]
default_yrange = [y_min - pad_y, y_max + pad_y]


# ---------------------------------------------------------------------------
# Title + Controls (left column)
# ---------------------------------------------------------------------------
st.title("Phase 1 Prototype — Genre Map")
col_left, col_main = st.columns([0.33, 0.67], gap="large")

with col_left:
    st.subheader("Controls")
    search = st.text_input("Search genre", value="")
    if search.strip():
        filtered = [g for g in genres if search.lower() in g.lower()] or genres
    else:
        filtered = genres

    options = ["(none)"] + filtered
    cur     = st.session_state.selected_genre
    if cur != "(none)" and cur not in options and cur in genres:
        options = ["(none)", cur] + filtered

    st.selectbox("Selected genre", options, key="selected_genre")
    # Sync _active_genre from the dropdown (runs on every full rerun)
    sel_from_dropdown = st.session_state.selected_genre
    if sel_from_dropdown != "(none)":
        st.session_state._active_genre = sel_from_dropdown
        push_history(sel_from_dropdown)
    elif st.session_state._pending_click is None:
        # Only clear _active_genre if no pending click
        st.session_state._active_genre = ""

    st.markdown("---")

    if enable_path:
        st.subheader("Shortest path finder")
        dest_options = ["(none)"] + genres
        st.selectbox("Destination genre", dest_options, key="destination_genre")
        dest = "" if st.session_state.destination_genre == "(none)" else st.session_state.destination_genre

        cols = st.columns([1, 1])
        if cols[0].button("Find shortest path", use_container_width=True):
            sel_g = st.session_state._active_genre
            if not sel_g or not dest:
                st.warning("Pick both a start genre and a destination genre.")
            elif sel_g not in genre_to_idx or dest not in genre_to_idx:
                st.warning("One of the chosen genres isn't in the dataset.")
            else:
                p = shortest_path(adj, genre_to_idx[sel_g], genre_to_idx[dest])
                if not p:
                    st.warning("No path found (graph disconnected for these two).")
                    st.session_state.active_path = []
                else:
                    st.session_state.active_path = p
                    push_history(dest)

        if cols[1].button("Clear path", use_container_width=True):
            st.session_state.active_path = []
            st.rerun()


# ---------------------------------------------------------------------------
# Map figure builder (called from within the fragment)
# ---------------------------------------------------------------------------
def build_map_figure(active: str) -> Tuple[go.Figure, np.ndarray]:
    fade       = clamp(fade_others / 100.0, 0.0, 0.95)
    ls         = clamp(line_strength / 100.0, 0.0, 1.0)
    active_idxs = None

    if active and active in genre_to_idx:
        si          = genre_to_idx[active]
        active_idxs = set(adj[si]) | {si}

    if st.session_state.active_path:
        active_idxs = set(st.session_state.active_path)

    base_rgb = df[["r", "g", "b"]].to_numpy(dtype=int)
    rgba     = []
    for i in range(len(df)):
        r, g, b = base_rgb[i]
        a = 1.0 if (active_idxs is None or i in active_idxs) else (1.0 - fade)
        rgba.append(f"rgba({r},{g},{b},{a})")

    sizes = np.full(len(df), dot_size, dtype=float)
    if active and active in genre_to_idx:
        sizes[genre_to_idx[active]] = dot_size + 4
    for i in (st.session_state.active_path or []):
        if 0 <= i < len(sizes):
            sizes[i] = max(sizes[i], dot_size + 3)

    line_traces: List[go.Scatter] = []
    if show_lines:
        if active_idxs is None:
            xs, ys = [], []
            for i in range(len(df)):
                for j in neighbors[i]:
                    xs += [df.at[i, "x"], df.at[j, "x"], None]
                    ys += [df.at[i, "y"], df.at[j, "y"], None]
            line_traces.append(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=1, color=f"rgba(220,220,220,{0.18 + 0.18*ls})"),
                hoverinfo="skip", name="edges", showlegend=False,
            ))
        else:
            xs, ys     = [], []
            active_set = set(active_idxs)
            for i in active_set:
                for j in adj[i]:
                    if j in active_set and j >= i:
                        xs += [df.at[i, "x"], df.at[j, "x"], None]
                        ys += [df.at[i, "y"], df.at[j, "y"], None]
            line_traces.append(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=2, color=f"rgba(255,255,255,{0.25 + 0.45*ls})"),
                hoverinfo="skip", name="edges_focus", showlegend=False,
            ))

        path = st.session_state.active_path
        if path and len(path) >= 2:
            xs, ys = [], []
            for a, b in zip(path, path[1:]):
                xs += [df.at[a, "x"], df.at[b, "x"], None]
                ys += [df.at[a, "y"], df.at[b, "y"], None]
            line_traces.append(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=4, color="rgba(255,255,255,0.95)"),
                hoverinfo="skip", name="path", showlegend=False,
            ))

    marker_trace = go.Scatter(
        x=df["x"], y=df["y"],
        mode="markers",
        marker=dict(size=sizes, color=rgba, line=dict(width=0)),
        text=df["genre"],
        hovertemplate="%{text}<extra></extra>",
        name="markers", showlegend=False,
    )

    label_indices = label_idxs_default
    label_trace   = None
    active_label_trace = None
    if show_labels:
        label_indices = pick_label_points(df, max_labels=default_label_count)
        label_trace   = go.Scatter(
            x=df.loc[label_indices, "x"],
            y=df.loc[label_indices, "y"],
            mode="text",
            text=df.loc[label_indices, "genre"],
            textposition="top center",
            textfont=dict(size=label_size, color="rgba(255,255,255,0.8)"),
            hoverinfo="skip", name="labels", showlegend=False,
        )

    if active and active in genre_to_idx:
        ai = genre_to_idx[active]
        active_label_trace = go.Scatter(
            x=[df.at[ai, "x"]],
            y=[df.at[ai, "y"]],
            mode="text",
            text=[active],
            textposition="top center",
            textfont=dict(size=label_size + 1, color="rgba(255,255,255,1.0)"),
            hoverinfo="skip", name="active_label", showlegend=False,
        )

    fig = go.Figure()
    for tr in line_traces:
        fig.add_trace(tr)
    fig.add_trace(marker_trace)
    if label_trace is not None:
        fig.add_trace(label_trace)
    if active_label_trace is not None:
        fig.add_trace(active_label_trace)

    rev = f"map-v7-{map_fit}-{st.session_state.reset_view_tick}"

    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode=False,
        hovermode="closest",
        uirevision=rev,
    )
    fig.update_xaxes(
        visible=False, range=default_xrange,
        autorange=False, uirevision=rev,
    )
    fig.update_yaxes(
        visible=False, range=default_yrange,
        autorange=False, uirevision=rev,
    )

    if map_fit == "Original":
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig, label_indices


def clicked_genre_from_selection(
    fig: go.Figure, label_indices: np.ndarray, curve: int, pidx: int
) -> Optional[str]:
    if curve < 0 or curve >= len(fig.data):
        return None
    tr   = fig.data[curve]
    name = getattr(tr, "name", "") or ""
    if name == "markers":
        if 0 <= pidx < len(genres):
            return genres[pidx]
    if name == "labels":
        if 0 <= pidx < len(label_indices):
            gi = int(label_indices[pidx])
            if 0 <= gi < len(genres):
                return genres[gi]
    if name == "active_label":
        return st.session_state._active_genre or None
    return None


# ---------------------------------------------------------------------------
# FRAGMENT: Map + Genre details
#
# @st.fragment isolates this section from the rest of the page.  When
# on_select fires (dot click), ONLY this fragment re-executes — the page
# around it (sidebar, controls) stays put.  Crucially, the Plotly component
# is *updated* (new props) rather than *recreated* (unmount/mount), which
# lets uirevision preserve the user's zoom and pan state.
# ---------------------------------------------------------------------------
@st.fragment
def map_and_details():
    col_map, col_details = st.columns([0.65, 0.35], gap="large")

    # Read the current active genre (may have been set by dropdown or
    # previous fragment click)
    active = st.session_state._active_genre

    # ---- Map column -------------------------------------------------------
    with col_map:
        st.subheader("Map (click a dot)")
        fig, label_indices = build_map_figure(active)

        config = {
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        }

        plot_key = f"{PLOT_KEY}-{map_fit}-{st.session_state.reset_view_tick}"

        event = st.plotly_chart(
            fig,
            use_container_width=True,
            config=config,
            on_select="rerun",          # only reruns THIS fragment
            selection_mode=["points"],
            key=plot_key,
        )

        # Detect click from on_select event
        clicked_genre: Optional[str] = None
        try:
            pts = event.selection.points
            if pts and isinstance(pts, list):
                p0    = pts[0]
                curve = p0.get("curve_number")
                pidx  = p0.get("point_index")
                if curve is not None and pidx is not None:
                    clicked_genre = clicked_genre_from_selection(
                        fig, label_indices, int(curve), int(pidx)
                    )
        except Exception:
            clicked_genre = None

        # Apply click — update _active_genre immediately (no st.rerun!)
        if clicked_genre and clicked_genre != active:
            st.session_state._active_genre = clicked_genre
            st.session_state._pending_click = clicked_genre
            st.session_state.active_path = []
            push_history(clicked_genre)
            active = clicked_genre      # use for details panel below

        st.caption("Tip: scroll to zoom · click a dot to select · use toolbar to pan")

    # ---- Genre details column ---------------------------------------------
    with col_details:
        st.subheader("Genre details")

        if not active:
            st.write("Pick a genre from the dropdown or click a dot on the map.")
        else:
            st.markdown(f"### {active}")
            wiki_title = resolve_wiki_title_for_genre(active)

            if not wiki_title:
                st.warning("No mapped Wikipedia page for this genre yet.")
            else:
                info = {
                    "title": wiki_title, "paragraph": "",
                    "url": f"https://en.wikipedia.org/wiki/{wiki_title.replace(' ','_')}",
                }
                try:
                    info = wiki_intro_by_title(wiki_title)
                except Exception as e:
                    st.warning(f"Could not load Wikipedia intro: {e}")

                para = (info.get("paragraph") or "").strip()
                url  = (info.get("url") or "").strip()
                if para:
                    st.write(para)
                else:
                    st.caption("No summary available for this page.")
                if url:
                    st.link_button("Open Wikipedia article", url)

            st.markdown("---")
            token  = spotify_get_token()
            market = st.session_state.market

            if not token:
                st.info(
                    "Spotify credentials not found. "
                    "Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to Streamlit Secrets."
                )
            else:
                example_track = None
                try:
                    if wiki_title:
                        example_track = pick_spotify_example_from_wikipedia(
                            token, wiki_title, market
                        )
                except Exception:
                    example_track = None

                if example_track and example_track.get("id"):
                    track_id     = example_track["id"]
                    track_name   = example_track.get("name", "Example track")
                    artist_names = ", ".join(
                        [a.get("name", "") for a in example_track.get("artists", [])]
                    )
                    picked_from = example_track.get("_picked_from_artist")

                    if picked_from:
                        st.markdown(
                            f"**Example (artist mentioned in intro):** {picked_from}"
                        )
                    else:
                        st.markdown("**Example:**")
                    st.write(f"{track_name} — {artist_names}")
                    st.components.v1.html(
                        spotify_embed_html(track_id),
                        height=160,
                        scrolling=False,
                    )
                    open_url = example_track.get("external_urls", {}).get("spotify")
                    if open_url:
                        st.link_button("Open in Spotify", open_url)
                else:
                    st.caption("No Spotify example found for this genre right now.")

        st.markdown(
            "<div class='small-muted'>Data: EveryNoise-derived genre attributes "
            "(genre, x, y, colour)</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Render the fragment inside col_main
# ---------------------------------------------------------------------------
with col_main:
    map_and_details()
