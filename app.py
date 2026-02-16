import base64
import heapq
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
# Wikipedia (better matching)
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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_wikipedia_intro(genre_name: str) -> Dict[str, Optional[str]]:
    headers = {"User-Agent": WIKI_UA}
    g = (genre_name or "").strip()
    if not g:
        return {"title": None, "paragraph": None, "url": None}

    banned = {
        "music genre",
        "music",
        "genre",
        "popular music",
        "list of genres",
        "outline of music",
    }

    # 1) Try exact title first
    try:
        direct = _fetch_page_intro_by_title(g)
        if direct.get("title") and norm_title(direct["title"]) not in banned and direct.get("paragraph"):
            return direct
    except Exception:
        pass

    # 2) Search several candidates and score them
    queries = [
        f'intitle:"{g}"',
        f'"{g}"',
        f"{g} music",
        f"{g} genre",
    ]

    candidates: List[str] = []
    for q in queries:
        try:
            r = requests.get(
                WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": q,
                    "srlimit": 5,
                    "format": "json",
                },
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

    # De-dupe while preserving order
    seen = set()
    cand_titles = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            cand_titles.append(t)

    ng = norm_title(g)

    def score_title(t: str) -> int:
        nt = norm_title(t)
        if nt in banned:
            return -9999
        if nt == ng:
            return 100
        if ng and ng in nt:
            return 80
        # token overlap
        toks = ng.split()
        if toks and all(tok in nt for tok in toks):
            return 60
        return 10

    best = {"title": None, "paragraph": None, "url": None}
    best_score = -9999

    for t in cand_titles[:12]:
        s = score_title(t)
        if s <= best_score:
            continue
        try:
            page = _fetch_page_intro_by_title(t)
            if not page.get("paragraph"):
                continue
            # extra penalty if paragraph is clearly generic
            pt = norm_title(page.get("title") or "")
            if pt in banned:
                continue
            best = page
            best_score = s
        except Exception:
            continue

    return best


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


def spotify_example_for_genre(genre_name: str, market: str = "AU") -> Optional[Dict[str, str]]:
    cache = st.session_state.setdefault("spotify_example_cache", {})
    key = (genre_name, market)
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
    if item_type not in {"track", "playlist"}:
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
# Plotly figure (smooth + readable)
# =========================
def make_figure(
    df_plot: pd.DataFrame,
    selected_idx: Optional[int],
    neighbor_ordered: List[int],
    edges_all: List[Tuple[int, int, float]],
    adj: List[List[Tuple[int, float]]],
    show_edges: bool,
    path: List[int],
    show_labels: bool,
    neighbor_label_count: int,
    show_hover: bool,
) -> go.Figure:
    n = len(df_plot)
    fig = go.Figure()

    has_selection = selected_idx is not None
    has_path = len(path) >= 2

    # Common arrays
    X = df_plot["x"].astype(float).to_numpy()
    Y = df_plot["y"].astype(float).to_numpy()
    C = df_plot["hex_colour"].to_numpy()
    G = df_plot["genre"].to_numpy()
    IDX = np.arange(n)

    # --- OVERVIEW MODE (no selection) ---
    if not has_selection:
        if show_edges and edges_all:
            xs, ys = [], []
            for u, v, _d in edges_all:
                xs.extend([float(X[u]), float(X[v]), None])
                ys.extend([float(Y[u]), float(Y[v]), None])
            fig.add_trace(
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(width=1.2, color="rgba(150,180,255,0.35)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scattergl(
                x=X,
                y=Y,
                mode="markers",
                marker=dict(size=6, color=C, opacity=0.92),
                text=G if show_hover else None,
                customdata=IDX,
                hovertemplate="<b>%{text}</b><extra></extra>" if show_hover else None,
                hoverinfo="text" if show_hover else "skip",
                showlegend=False,
            )
        )

    # --- SELECTION / PATH MODE ---
    else:
        # Background points (faint)
        bg_opacity = 0.25 if not has_path else 0.14
        fig.add_trace(
            go.Scattergl(
                x=X,
                y=Y,
                mode="markers",
                marker=dict(size=5, color=C, opacity=bg_opacity),
                customdata=IDX,
                text=G if show_hover else None,
                hovertemplate="<b>%{text}</b><extra></extra>" if show_hover else None,
                hoverinfo="text" if show_hover else "skip",
                showlegend=False,
            )
        )

        if show_edges:
            if has_path:
                # Path line
                px, py = [], []
                for a, b in zip(path[:-1], path[1:]):
                    px.extend([float(X[a]), float(X[b]), None])
                    py.extend([float(Y[a]), float(Y[b]), None])
                fig.add_trace(
                    go.Scattergl(
                        x=px,
                        y=py,
                        mode="lines",
                        line=dict(width=4.0, color="rgba(120,200,255,0.92)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            else:
                # Selected -> neighbor star
                sx, sy = float(X[selected_idx]), float(Y[selected_idx])
                xs, ys = [], []
                for v, _d in adj[selected_idx]:
                    xs.extend([sx, float(X[v]), None])
                    ys.extend([sy, float(Y[v]), None])
                fig.add_trace(
                    go.Scattergl(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(width=3.2, color="rgba(255,255,255,0.75)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # Highlighted points
        if has_path:
            path_set = sorted(set([i for i in path if 0 <= i < n]))
            # Path nodes
            fig.add_trace(
                go.Scattergl(
                    x=X[path_set],
                    y=Y[path_set],
                    mode="markers",
                    marker=dict(size=10, color=C[path_set], opacity=1.0),
                    customdata=IDX[path_set],
                    text=G[path_set] if show_hover else None,
                    hovertemplate="<b>%{text}</b><extra></extra>" if show_hover else None,
                    hoverinfo="text" if show_hover else "skip",
                    showlegend=False,
                )
            )
            # Emphasize endpoints
            endpoints = [path[0], path[-1]]
            endpoints = [i for i in endpoints if 0 <= i < n]
            if endpoints:
                fig.add_trace(
                    go.Scattergl(
                        x=X[endpoints],
                        y=Y[endpoints],
                        mode="markers",
                        marker=dict(size=14, color=C[endpoints], opacity=1.0),
                        customdata=IDX[endpoints],
                        text=G[endpoints] if show_hover else None,
                        hovertemplate="<b>%{text}</b><extra></extra>" if show_hover else None,
                        hoverinfo="text" if show_hover else "skip",
                        showlegend=False,
                    )
                )
        else:
            # Neighbors
            neigh = neighbor_ordered
            if neigh:
                fig.add_trace(
                    go.Scattergl(
                        x=X[neigh],
                        y=Y[neigh],
                        mode="markers",
                        marker=dict(size=9, color=C[neigh], opacity=1.0),
                        customdata=IDX[neigh],
                        text=G[neigh] if show_hover else None,
                        hovertemplate="<b>%{text}</b><extra></extra>" if show_hover else None,
                        hoverinfo="text" if show_hover else "skip",
                        showlegend=False,
                    )
                )
            # Selected point
            fig.add_trace(
                go.Scattergl(
                    x=[float(X[selected_idx])],
                    y=[float(Y[selected_idx])],
                    mode="markers",
                    marker=dict(size=14, color=[str(C[selected_idx])], opacity=1.0),
                    customdata=[int(selected_idx)],
                    text=[str(G[selected_idx])] if show_hover else None,
                    hovertemplate="<b>%{text}</b><extra></extra>" if show_hover else None,
                    hoverinfo="text" if show_hover else "skip",
                    showlegend=False,
                )
            )

        # Labels (limited so they stay readable)
        if show_labels:
            label_idxs: List[int] = []
            if has_path:
                # Start/end only (always readable)
                label_idxs = [path[0], path[-1]]
                label_idxs = [i for i in label_idxs if 0 <= i < n]
            else:
                # Selected + a few "spread out" neighbors (we take the farthest ones)
                label_idxs = [selected_idx]
                if neighbor_label_count > 0 and neighbor_ordered:
                    # neighbor_ordered is farthest-first already
                    label_idxs += neighbor_ordered[:neighbor_label_count]

            label_idxs = sorted(set(label_idxs))
            if label_idxs:
                fig.add_trace(
                    go.Scatter(
                        x=[float(X[i]) for i in label_idxs],
                        y=[float(Y[i]) for i in label_idxs],
                        mode="text",
                        text=[str(G[i]) for i in label_idxs],
                        textposition="top center",
                        textfont=dict(size=12, color="rgba(255,255,255,0.92)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    fig.update_layout(
        template="plotly_dark",
        height=720,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        dragmode="pan",
        clickmode="event+select",
        hovermode="closest" if show_hover else False,
        uirevision="keep",
    )
    return fig


# =========================
# App UI
# =========================
st.set_page_config(page_title="Phase 1: Genre Map", layout="wide")

st.markdown(
    """
<style>
div[data-baseweb="select"] * { cursor: pointer !important; }
div[data-baseweb="select"] input { caret-color: transparent; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Phase 1 Prototype — Genre Map")

# Default state: no selection on first load
if "selected_genre_dropdown" not in st.session_state:
    st.session_state["selected_genre_dropdown"] = NONE_OPTION

if "current_path" not in st.session_state:
    st.session_state["current_path"] = []
if "path_message" not in st.session_state:
    st.session_state["path_message"] = ""

# Apply pending click selection BEFORE widgets are created (safe)
pending = st.session_state.pop("pending_genre", None)
if pending:
    st.session_state["selected_genre_dropdown"] = pending
    st.session_state["current_path"] = []
    st.session_state["path_message"] = ""

# Sidebar
with st.sidebar:
    st.header("Selection history")
    hist = st.session_state.get("genre_history", [])
    if not hist:
        st.caption("Click dots or pick genres to build history.")
    else:
        for i, g in enumerate(hist[:12]):
            if st.button(g, key=f"hist_{i}", use_container_width=True):
                st.session_state["selected_genre_dropdown"] = g
                st.session_state["current_path"] = []
                st.session_state["path_message"] = ""
                st.rerun()
        cols = st.columns(2)
        with cols[0]:
            if st.button("Clear", use_container_width=True):
                st.session_state["genre_history"] = []
                st.session_state["selected_genre_dropdown"] = NONE_OPTION
                st.session_state["current_path"] = []
                st.session_state["path_message"] = ""
                st.rerun()
        with cols[1]:
            st.caption(f"{len(hist)} saved")

    st.divider()

    st.header("Data")
    source = st.radio("Load dataset from:", ["Web (recommended)", "Upload CSV"], index=0)
    uploaded_bytes = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload genre CSV", type=["csv"])
        if not uploaded:
            st.stop()
        uploaded_bytes = uploaded.getvalue()

    st.divider()
    st.header("Connections")
    k = st.slider("Connections per genre", 2, 20, 8)
    st.checkbox("Show connection lines", value=True, key="show_edges")
    st.checkbox("Enable path finder", value=True, key="enable_path")

    st.divider()
    st.header("Labels")
    show_labels = st.checkbox("Show map labels", value=True)
    neighbor_label_count = st.slider("Neighbor labels (when selected)", 0, 20, 6)
    show_hover = st.checkbox("Show hover popups (can block text)", value=False)

    st.divider()
    st.header("Spotify examples")
    market = st.selectbox("Country/market", ["AU", "US", "GB", "CA", "NZ", "DE", "FR"], index=0)

    st.divider()
    st.header("View")
    view_mode = st.radio("Map shape", ["Fit to screen", "Original"], index=0)

# Load data
df = load_genre_data("upload" if source == "Upload CSV" else "web", uploaded_bytes)

# Prepare plot coords
df_plot = df.copy()
if view_mode == "Fit to screen":
    x_min, x_max = df_plot["x"].min(), df_plot["x"].max()
    y_min, y_max = df_plot["y"].min(), df_plot["y"].max()
    if x_max != x_min:
        df_plot["x"] = (df_plot["x"] - x_min) / (x_max - x_min)
    if y_max != y_min:
        df_plot["y"] = (df_plot["y"] - y_min) / (y_max - y_min)

coords = df_plot[["x", "y"]].to_numpy(dtype=float)
adj, undirected_edges = build_knn_graph(coords, k=k)

col_controls, col_map, col_details = st.columns([1.1, 2.2, 1.3], gap="large")

# ---------------- Controls column ----------------
with col_controls:
    st.subheader("Controls")

    q = st.text_input("Search genre", value="", key="start_query")
    if q.strip():
        mask = df["genre"].str.contains(q.strip(), case=False, na=False)
        candidates = df.loc[mask, "genre"].tolist()
    else:
        candidates = df["genre"].head(800).tolist()

    options = [NONE_OPTION] + candidates

    # IMPORTANT: no index= here (prevents the yellow Streamlit warning)
    chosen = st.selectbox("Selected genre", options, key="selected_genre_dropdown")

    # Clear path whenever selection changes
    if st.session_state.get("prev_dropdown") != chosen:
        st.session_state["prev_dropdown"] = chosen
        st.session_state["current_path"] = []
        st.session_state["path_message"] = ""

    if chosen != NONE_OPTION:
        push_history(chosen)
        selected_idx = int(df.index[df["genre"] == chosen][0])

        # neighbors sorted by distance (farthest-first for label spread)
        neigh_pairs = sorted(adj[selected_idx], key=lambda t: t[1], reverse=True)
        neighbor_ordered = [v for v, _d in neigh_pairs]
        neighbor_idxs = set(neighbor_ordered)

        neighbor_list = df.loc[list(neighbor_idxs), "genre"].sort_values().tolist()
        st.caption("Closest genres")
        st.write(", ".join(neighbor_list[:25]) + (" ..." if len(neighbor_list) > 25 else ""))
    else:
        selected_idx = None
        neighbor_ordered = []
        st.caption("Closest genres")
        st.write("Select a genre to show its nearest neighbors.")

    if st.session_state.get("enable_path", True):
        st.markdown("### Path finder")
        if chosen == NONE_OPTION:
            st.info("Pick a start genre first, then you can find a path.")
        else:
            dest_q = st.text_input("Search destination", value="", key="dest_query")
            if dest_q.strip():
                dest_mask = df["genre"].str.contains(dest_q.strip(), case=False, na=False)
                end_candidates = df.loc[dest_mask, "genre"].tolist()
            else:
                end_candidates = stable_sample_genres(df["genre"], n=500)

            if end_candidates:
                end = st.selectbox("Destination genre", end_candidates, index=0, key="dest_genre")
                if st.button("Find shortest path"):
                    end_idx = int(df.index[df["genre"] == end][0])
                    path = dijkstra_path(adj, selected_idx, end_idx)
                    st.session_state["current_path"] = path
                    st.session_state["path_message"] = (
                        f"Path found: {len(path)} steps." if path else "No path found. Try increasing connections."
                    )
                    st.rerun()

    if st.session_state.get("path_message"):
        if st.session_state["current_path"]:
            st.success(st.session_state["path_message"])
        else:
            st.error(st.session_state["path_message"])

# ---------------- Map column ----------------
with col_map:
    st.subheader("Map (click a dot)")

    fig = make_figure(
        df_plot=df_plot,
        selected_idx=selected_idx,
        neighbor_ordered=neighbor_ordered,
        edges_all=undirected_edges,
        adj=adj,
        show_edges=st.session_state.get("show_edges", True),
        path=st.session_state.get("current_path", []),
        show_labels=show_labels,
        neighbor_label_count=neighbor_label_count,
        show_hover=show_hover,
    )

    event = st.plotly_chart(
        fig,
        key="genre_map",
        on_select="rerun",
        selection_mode="points",
        config={"scrollZoom": True, "doubleClick": "reset", "displaylogo": False, "responsive": True},
    )

    # Click selection: use customdata so it works even with multiple marker traces
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")

    if sel:
        points = getattr(sel, "points", None)
        if points is None and isinstance(sel, dict):
            points = sel.get("points")

        if points and len(points) > 0:
            p0 = points[0]
            idx = p0.get("customdata")  # original row index
            if idx is not None:
                idx = int(idx)
                if 0 <= idx < len(df) and st.session_state.get("last_click_idx") != idx:
                    st.session_state["last_click_idx"] = idx
                    clicked_genre = df.loc[idx, "genre"]
                    st.session_state["pending_genre"] = clicked_genre
                    push_history(clicked_genre)
                    st.rerun()

# ---------------- Details column ----------------
with col_details:
    st.subheader("Genre details")

    genre_name = st.session_state.get("selected_genre_dropdown", NONE_OPTION)
    if genre_name == NONE_OPTION:
        st.caption("Click a dot or choose a genre to see details.")
    else:
        st.markdown(f"**{genre_name}**")

        wiki = fetch_wikipedia_intro(genre_name)
        if wiki.get("paragraph"):
            st.write(wiki["paragraph"])
        else:
            st.caption("No clear Wikipedia summary found for this genre name.")

        if wiki.get("url"):
            st.markdown(f"[Open Wikipedia article]({wiki['url']})")

        st.divider()

        client_id, client_secret = _get_spotify_creds()
        if not client_id or not client_secret:
            st.warning("Spotify keys not found in Streamlit Secrets.")
            st.caption("Manage app → Settings → Secrets")
        else:
            ex = spotify_example_for_genre(genre_name, market=market)
            if ex:
                st.markdown(f"**Example:** {ex['name']} — {ex['subtitle']}")
                st.components.v1.html(spotify_embed_html(ex["type"], ex["id"], height=152), height=170, scrolling=False)
                if ex.get("url"):
                    st.markdown(f"[Open in Spotify]({ex['url']})")
            else:
                st.caption("Couldn’t find a Spotify example for this exact genre name.")
                st.markdown(f"[Search this in Spotify](https://open.spotify.com/search/{quote(genre_name)})")
