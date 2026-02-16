import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ----------------------------
# Page setup + CSS tweaks
# ----------------------------
st.set_page_config(page_title="Phase 1 Prototype — Genre Map", layout="wide")

st.markdown(
    """
<style>
/* Make dropdown feel clickable (cursor pointer) */
div[data-baseweb="select"] * { cursor: pointer !important; }

/* Reduce chart padding */
div[data-testid="stPlotlyChart"] > div { padding: 0 !important; }

/* Sidebar spacing */
section[data-testid="stSidebar"] .block-container { padding-top: 1.0rem; }

/* Subtle muted text */
.small-muted { font-size: 0.85rem; color: rgba(255,255,255,0.65); }
</style>
""",
    unsafe_allow_html=True,
)

UA = "genre-map-proto/1.0 (personal project)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API = "https://api.spotify.com/v1"

# Keep Plotly view stable across reruns (prevents “map reset”)
UIREV = "genre-map-uirev-2"
PLOT_KEY = "genre_map_plot"


# ----------------------------
# Helpers
# ----------------------------
def norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_get_secret(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return (os.environ.get(name) or "").strip() or None


def http_get(url: str, params: dict, timeout: int = 30) -> dict:
    headers = {"User-Agent": UA}
    for attempt in range(6):
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            sleep_s = int(retry) if retry and retry.isdigit() else (1 + attempt * 2)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Too many requests for {url}")


# ----------------------------
# Spotify helpers
# ----------------------------
def spotify_get_token() -> Optional[str]:
    client_id = safe_get_secret("SPOTIFY_CLIENT_ID")
    client_secret = safe_get_secret("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {"grant_type": "client_credentials"}
    headers = {"User-Agent": UA}

    for attempt in range(6):
        r = requests.post(SPOTIFY_TOKEN_URL, data=data, auth=auth, headers=headers, timeout=30)
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            sleep_s = int(retry) if retry and retry.isdigit() else (1 + attempt * 2)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
        return r.json().get("access_token")
    return None


def spotify_api_get(path: str, token: str, params: dict | None = None, timeout: int = 30) -> dict:
    headers = {"Authorization": f"Bearer {token}", "User-Agent": UA}
    url = f"{SPOTIFY_API}{path}"
    for attempt in range(6):
        r = requests.get(url, params=params or {}, headers=headers, timeout=timeout)
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            sleep_s = int(retry) if retry and retry.isdigit() else (1 + attempt * 2)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Spotify GET failed: {path}")


def spotify_search_artist(token: str, name: str, market: str) -> Optional[dict]:
    js = spotify_api_get(
        "/search",
        token,
        params={"q": name, "type": "artist", "limit": 5, "market": market},
        timeout=30,
    )
    items = js.get("artists", {}).get("items", [])
    if not items:
        return None

    n = norm_title(name)
    for a in items:
        if norm_title(a.get("name", "")) == n:
            return a
    return items[0]


def spotify_top_track_for_artist(token: str, artist_id: str, market: str) -> Optional[dict]:
    js = spotify_api_get(f"/artists/{artist_id}/top-tracks", token, params={"market": market}, timeout=30)
    tracks = js.get("tracks", [])
    return tracks[0] if tracks else None


def spotify_search_track(token: str, query: str, market: str) -> Optional[dict]:
    js = spotify_api_get(
        "/search",
        token,
        params={"q": query, "type": "track", "limit": 5, "market": market},
        timeout=30,
    )
    items = js.get("tracks", {}).get("items", [])
    return items[0] if items else None


def spotify_embed_html(track_id: str) -> str:
    return f"""
    <iframe style="border-radius:12px"
      src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
      width="100%" height="152" frameborder="0"
      allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
      loading="lazy"></iframe>
    """


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_genre_data_from_repo() -> pd.DataFrame:
    base = Path(__file__).resolve().parent
    path = base / "data" / "genre_attrs.csv"
    if not path.exists():
        raise FileNotFoundError("Missing data/genre_attrs.csv in the repo.")

    df = pd.read_csv(path)
    colmap = {c.lower(): c for c in df.columns}

    def pick(name: str) -> Optional[str]:
        return colmap.get(name)

    genre_col = pick("genre") or pick("name")
    x_col = pick("x")
    y_col = pick("y")

    if not genre_col or not x_col or not y_col:
        raise ValueError("CSV must contain at least: genre, x, y columns.")

    df = df.rename(columns={genre_col: "genre", x_col: "x", y_col: "y"})

    r_col, g_col, b_col = pick("r"), pick("g"), pick("b")
    if r_col and g_col and b_col:
        df = df.rename(columns={r_col: "r", g_col: "g", b_col: "b"})
    else:
        df["r"], df["g"], df["b"] = 180, 180, 180

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["r"] = pd.to_numeric(df["r"], errors="coerce").fillna(180).astype(int)
    df["g"] = pd.to_numeric(df["g"], errors="coerce").fillna(180).astype(int)
    df["b"] = pd.to_numeric(df["b"], errors="coerce").fillna(180).astype(int)

    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    df["genre"] = df["genre"].astype(str)

    df["hex"] = df.apply(lambda row: f"rgb({int(row.r)},{int(row.g)},{int(row.b)})", axis=1)
    return df


@st.cache_data(show_spinner=False)
def load_wiki_title_maps() -> tuple[dict[str, str], dict[str, str]]:
    base = Path(__file__).resolve().parent
    overrides_path = base / "data" / "wiki_overrides.csv"
    map_path = base / "data" / "wiki_map.csv"

    overrides: dict[str, str] = {}
    generated: dict[str, str] = {}

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
                if g and t:
                    generated[g] = t

    return overrides, generated


# ----------------------------
# Wikipedia helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def wiki_intro_by_title(title: str) -> dict:
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
    t = page.get("title") or title
    extract = (page.get("extract") or "").strip()
    paragraph = extract.split("\n")[0].strip() if extract else ""
    url = f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}" if t else ""
    return {"title": t, "paragraph": paragraph, "url": url}


@st.cache_data(show_spinner=False)
def wiki_lead_html(title: str) -> str:
    js = http_get(
        WIKI_API,
        params={
            "action": "parse",
            "page": title,
            "prop": "text",
            "section": 0,
            "format": "json",
        },
        timeout=30,
    )
    return (js.get("parse", {}).get("text", {}).get("*") or "")


def extract_first_paragraph_link_titles(html: str, max_items: int = 25) -> List[str]:
    if not html:
        return []
    m = re.search(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    p = m.group(1)

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


def resolve_wiki_title_for_genre(genre: str) -> Optional[str]:
    overrides_map, generated_map = load_wiki_title_maps()
    key = norm_title(genre)
    return overrides_map.get(key) or generated_map.get(key)


def pick_spotify_example_from_wikipedia(token: Optional[str], wiki_title: str, market: str) -> Optional[dict]:
    if not token:
        return None

    try:
        html = wiki_lead_html(wiki_title)
        link_titles = extract_first_paragraph_link_titles(html, max_items=25)
    except Exception:
        link_titles = []

    # Filter obvious non-artist links (best-effort)
    bad_words = [
        "music", "genre", "style", "history", "list", "album", "song", "record",
        "city", "state", "country", "united states", "england", "australia"
    ]
    filtered = []
    for t in link_titles:
        tl = norm_title(t)
        if any(bw in tl for bw in bad_words):
            continue
        if len(t) > 45:
            continue
        filtered.append(t)

    for name in filtered[:12]:
        try:
            artist = spotify_search_artist(token, name, market=market)
            if artist and artist.get("id"):
                top = spotify_top_track_for_artist(token, artist["id"], market=market)
                if top and top.get("id"):
                    top["_picked_from_artist"] = artist.get("name")
                    return top
        except Exception:
            continue

    # Fallback: track search by title
    try:
        tr = spotify_search_track(token, wiki_title, market=market)
        if tr and tr.get("id"):
            return tr
    except Exception:
        pass

    return None


# ----------------------------
# Neighbors + path
# ----------------------------
@st.cache_data(show_spinner=False)
def compute_neighbors(xy: np.ndarray, k: int) -> List[List[int]]:
    n = xy.shape[0]
    k = int(clamp(k, 1, 25))

    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto")
        nn.fit(xy)
        _, idxs = nn.kneighbors(xy, return_distance=True)
        return [list(row[1:]) for row in idxs]
    except Exception:
        pass

    xy_f = xy.astype(np.float32)
    out: List[List[int]] = []
    for i in range(n):
        d = np.sum((xy_f - xy_f[i]) ** 2, axis=1)
        idx = np.argpartition(d, min(k + 1, n - 1))[: min(k + 1, n)]
        idx = idx[idx != i]
        idx = idx[np.argsort(d[idx])][:k]
        out.append(idx.astype(int).tolist())
    return out


def build_adjacency(neighbors: List[List[int]]) -> List[List[int]]:
    n = len(neighbors)
    adj = [[] for _ in range(n)]
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if j not in adj[i]:
                adj[i].append(j)
            if i not in adj[j]:
                adj[j].append(i)
    return adj


def shortest_path(adj: List[List[int]], start: int, goal: int) -> List[int]:
    if start == goal:
        return [start]
    n = len(adj)
    prev = [-1] * n
    q = deque([start])
    prev[start] = start

    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if prev[nxt] != -1:
                continue
            prev[nxt] = cur
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)

    if prev[goal] == -1:
        return []

    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# ----------------------------
# Label selection (avoid “white blob”)
# ----------------------------
@st.cache_data(show_spinner=False)
def pick_label_points(df: pd.DataFrame, max_labels: int = 90) -> np.ndarray:
    max_labels = int(clamp(max_labels, 30, 300))
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    # Grid sampling
    bins = int(np.sqrt(max_labels) * 1.5)
    bins = int(clamp(bins, 8, 45))

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    gx = np.floor((x - x_min) / (x_max - x_min + 1e-9) * bins).astype(int)
    gy = np.floor((y - y_min) / (y_max - y_min + 1e-9) * bins).astype(int)

    chosen = {}
    for idx in range(len(df)):
        key = (gx[idx], gy[idx])
        if key not in chosen:
            chosen[key] = idx
        if len(chosen) >= max_labels:
            break

    return np.array(list(chosen.values()), dtype=int)


# ----------------------------
# Session state
# ----------------------------
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = "(none)"
if "destination_genre" not in st.session_state:
    st.session_state.destination_genre = "(none)"
if "selection_history" not in st.session_state:
    st.session_state.selection_history = []
if "active_path" not in st.session_state:
    st.session_state.active_path = []
if "market" not in st.session_state:
    st.session_state.market = "AU"
if "reset_view_tick" not in st.session_state:
    st.session_state.reset_view_tick = 0  # changes uirevision when user wants a reset


def push_history(genre: str, max_items: int = 12) -> None:
    g = (genre or "").strip()
    if not g or g == "(none)":
        return
    hist = st.session_state.selection_history
    if g in hist:
        hist.remove(g)
    hist.insert(0, g)
    st.session_state.selection_history = hist[:max_items]


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## Selection history")
    st.caption("Click dots or pick genres to build history.")
    if st.session_state.selection_history:
        for g in st.session_state.selection_history:
            if st.button(g, use_container_width=True):
                st.session_state.selected_genre = g
                st.session_state.active_path = []
                push_history(g)
                st.rerun()
        cols = st.columns([1, 1])
        if cols[0].button("Clear", use_container_width=True):
            st.session_state.selection_history = []
            st.rerun()
        cols[1].markdown(f"<div class='small-muted'>{len(st.session_state.selection_history)} saved</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-muted'>No history yet.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Data")
    st.caption("Genres come from your EveryNoise-derived CSV: data/genre_attrs.csv")
    data_source = st.radio("Load dataset from:", ["Repo CSV (recommended)", "Upload CSV"], index=0)
    uploaded = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    st.markdown("## Connections")
    k = st.slider("Connections per genre (nearest neighbors)", min_value=2, max_value=20, value=8, step=1)
    show_lines = st.checkbox("Show connection lines", value=True)
    enable_path = st.checkbox("Enable path finder", value=True)

    st.markdown("---")
    st.markdown("## Labels")
    show_labels = st.checkbox("Show labels on map", value=True)
    default_label_count = st.slider("Default label count", 30, 220, 80, 10)
    label_size = st.slider("Label size", 8, 16, 10, 1)

    st.markdown("---")
    st.markdown("## Visibility")
    fade_others = st.slider("Fade non-selected dots", 0, 90, 65, 5)  # % fade
    line_strength = st.slider("Line strength", 0, 100, 55, 5)       # %
    dot_size = st.slider("Dot size", 4, 12, 7, 1)

    st.markdown("---")
    st.markdown("## Spotify examples")
    market = st.selectbox("Country/market", ["AU", "US", "GB", "CA", "NZ", "DE", "FR", "BR", "JP"])
    st.session_state.market = market

    st.markdown("---")
    st.markdown("## View")
    map_fit = st.radio("Map shape", ["Fit to screen", "Original"], index=0)
    if st.button("Reset map view", use_container_width=True):
        st.session_state.reset_view_tick += 1
        st.rerun()


# ----------------------------
# Load data
# ----------------------------
try:
    if data_source == "Upload CSV" and uploaded is not None:
        df = pd.read_csv(uploaded)
        colmap = {c.lower(): c for c in df.columns}
        if "genre" not in colmap or "x" not in colmap or "y" not in colmap:
            st.error("Uploaded CSV must include columns: genre, x, y (and optionally r,g,b).")
            df = None
        else:
            df = df.rename(columns={colmap["genre"]: "genre", colmap["x"]: "x", colmap["y"]: "y"})
            if all(c in colmap for c in ["r", "g", "b"]):
                df = df.rename(columns={colmap["r"]: "r", colmap["g"]: "g", colmap["b"]: "b"})
            else:
                df["r"], df["g"], df["b"] = 180, 180, 180
            df["x"] = pd.to_numeric(df["x"], errors="coerce")
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
            df["r"] = pd.to_numeric(df["r"], errors="coerce").fillna(180).astype(int)
            df["g"] = pd.to_numeric(df["g"], errors="coerce").fillna(180).astype(int)
            df["b"] = pd.to_numeric(df["b"], errors="coerce").fillna(180).astype(int)
            df["hex"] = df.apply(lambda row: f"rgb({int(row.r)},{int(row.g)},{int(row.b)})", axis=1)
    else:
        df = load_genre_data_from_repo()

    if df is None or df.empty:
        raise ValueError("No data loaded.")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

genres = df["genre"].tolist()
genre_to_idx = {g: i for i, g in enumerate(genres)}
xy = df[["x", "y"]].to_numpy(dtype=np.float32)

neighbors = compute_neighbors(xy, k=k)
adj = build_adjacency(neighbors)

label_idxs_default = pick_label_points(df, max_labels=default_label_count)

# ranges (used for reset + default)
x_min, x_max = float(df["x"].min()), float(df["x"].max())
y_min, y_max = float(df["y"].min()), float(df["y"].max())
pad_x = (x_max - x_min) * 0.04
pad_y = (y_max - y_min) * 0.04
default_xrange = [x_min - pad_x, x_max + pad_x]
default_yrange = [y_min - pad_y, y_max + pad_y]

# ----------------------------
# Main layout
# ----------------------------
st.title("Phase 1 Prototype — Genre Map")

col_left, col_mid, col_right = st.columns([0.33, 0.44, 0.23], gap="large")

# ----------------------------
# Controls panel (left main)
# ----------------------------
with col_left:
    st.subheader("Controls")

    search = st.text_input("Search genre", value="")

    if search.strip():
        filtered = [g for g in genres if search.lower() in g.lower()]
        if not filtered:
            filtered = genres
    else:
        filtered = genres

    options = ["(none)"] + filtered

    # Keep current selection available even if search filters it out
    cur = st.session_state.selected_genre
    if cur != "(none)" and cur not in options and cur in genres:
        options = ["(none)", cur] + filtered

    st.session_state.selected_genre = st.selectbox(
        "Selected genre",
        options,
        index=options.index(cur) if cur in options else 0,
        key="selected_genre_widget",
    )

    # internal selected value
    selected_genre = "" if st.session_state.selected_genre == "(none)" else st.session_state.selected_genre
    if selected_genre:
        push_history(selected_genre)

    # Neighbors list
    if selected_genre and selected_genre in genre_to_idx:
        si = genre_to_idx[selected_genre]
        close = [genres[j] for j in neighbors[si]]
        st.markdown("**Closest genres**")
        st.write(", ".join(close))

    # Path finder
    st.subheader("Path finder")
    if enable_path and selected_genre and selected_genre in genre_to_idx:
        dest_search = st.text_input("Search destination", value="", key="dest_search")
        dflt = dest_search.strip().lower()
        dest_filtered = [g for g in genres if dflt in g.lower()] if dflt else genres

        dest_options = ["(none)"] + dest_filtered

        cur_dest = st.session_state.destination_genre
        if cur_dest != "(none)" and cur_dest not in dest_options and cur_dest in genres:
            dest_options = ["(none)", cur_dest] + dest_filtered

        st.session_state.destination_genre = st.selectbox(
            "Destination genre",
            dest_options,
            index=dest_options.index(cur_dest) if cur_dest in dest_options else 0,
            key="dest_genre_widget",
        )

        destination = "" if st.session_state.destination_genre == "(none)" else st.session_state.destination_genre

        if st.button("Find shortest path", use_container_width=True):
            start_idx = genre_to_idx[selected_genre]
            if destination and destination in genre_to_idx:
                end_idx = genre_to_idx[destination]
                path = shortest_path(adj, start_idx, end_idx)
                st.session_state.active_path = path
                if path:
                    st.success(f"Path found: {len(path)} steps.")
                else:
                    st.warning("No path found (try increasing Connections per genre).")
            else:
                st.warning("Pick a destination genre first.")
    else:
        st.caption("Enable Path finder and select a genre to use this.")


# ----------------------------
# Plot building
# ----------------------------
def build_edges_trace(edge_pairs: List[Tuple[int, int]], color: str, width: float, opacity: float) -> go.Scatter:
    xs, ys = [], []
    for a, b in edge_pairs:
        xs.extend([df.at[a, "x"], df.at[b, "x"], None])
        ys.extend([df.at[a, "y"], df.at[b, "y"], None])
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity,
        hoverinfo="skip",
        showlegend=False,
    )


def build_map_figure() -> tuple[go.Figure, List[int]]:
    n = len(df)
    sel = selected_genre
    selected_idx = genre_to_idx.get(sel) if sel else None
    path_idxs = st.session_state.active_path or []

    neighbor_nodes = set()
    focus_nodes = set()
    if selected_idx is not None:
        focus_nodes.add(selected_idx)
        neighbor_nodes.update(neighbors[selected_idx])
    for p in path_idxs:
        focus_nodes.add(p)

    # Visibility tuning
    fade = fade_others / 100.0
    base_opacity = 0.95 if (selected_idx is None and not path_idxs) else (0.45 * (1 - fade) + 0.10)
    base_opacity = clamp(base_opacity, 0.15, 0.55)

    neighbor_opacity = 0.75
    focus_opacity = 0.98

    base_size = float(dot_size)
    neighbor_size = float(dot_size + 2)
    focus_size = float(dot_size + 4)

    marker_opacity = np.full(n, base_opacity, dtype=np.float32)
    marker_size = np.full(n, base_size, dtype=np.float32)

    if selected_idx is None and not path_idxs:
        marker_opacity[:] = 0.95
        marker_size[:] = base_size

    for idx in neighbor_nodes:
        marker_opacity[idx] = neighbor_opacity
        marker_size[idx] = neighbor_size

    for idx in focus_nodes:
        marker_opacity[idx] = focus_opacity
        marker_size[idx] = focus_size

    # Labels: avoid dense blob in default view
    label_indices: List[int] = []
    if show_labels:
        if selected_idx is None and not path_idxs:
            label_indices = label_idxs_default.tolist()
        else:
            # show selected, path, and a few neighbors
            label_set = set()
            if selected_idx is not None:
                label_set.add(selected_idx)
                for j in list(neighbor_nodes)[:18]:
                    label_set.add(j)
            for p in path_idxs:
                label_set.add(p)
            label_indices = sorted(label_set)

    # Build edges
    fig = go.Figure()

    if show_lines:
        strength = line_strength / 100.0
        # Default: show global edges (still reasonable for ~5k nodes)
        if selected_idx is None and not path_idxs:
            edges = []
            seen = set()
            for i in range(n):
                for j in neighbors[i]:
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) not in seen:
                        seen.add((a, b))
                        edges.append((a, b))
            # brighter than before
            fig.add_trace(build_edges_trace(edges, color=f"rgba(120,160,255,{0.30 + 0.50*strength})", width=1.0, opacity=0.70))
        else:
            # Local edges around focus
            edges_local = []
            seen = set()
            focus = set(focus_nodes) | set(neighbor_nodes)
            for i in focus:
                for j in neighbors[i]:
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) not in seen:
                        seen.add((a, b))
                        edges_local.append((a, b))
            if edges_local:
                fig.add_trace(build_edges_trace(edges_local, color=f"rgba(120,160,255,{0.45 + 0.45*strength})", width=1.2, opacity=0.85))

            if len(path_idxs) >= 2:
                path_edges = list(zip(path_idxs[:-1], path_idxs[1:]))
                fig.add_trace(build_edges_trace(path_edges, color="rgba(255,255,255,0.90)", width=2.6, opacity=0.95))

    # Markers (WebGL)
    fig.add_trace(
        go.Scattergl(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(
                size=marker_size.tolist(),
                color=df["hex"].tolist(),
                opacity=marker_opacity.tolist(),
                line=dict(width=0),
            ),
            hovertext=df["genre"],
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Labels trace (SVG), with correct click mapping
    if show_labels and label_indices:
        fig.add_trace(
            go.Scatter(
                x=df.loc[label_indices, "x"],
                y=df.loc[label_indices, "y"],
                mode="text",
                text=df.loc[label_indices, "genre"],
                textposition="top center",
                textfont=dict(size=label_size, color="rgba(255,255,255,0.88)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Layout:
    # - Fit to screen: DO NOT lock aspect ratio (prevents “thin vertical map” in narrow layouts)
    # - Original: lock aspect ratio
    uirev = f"{UIREV}-{st.session_state.reset_view_tick}"

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_dark",
        height=650,
        hovermode="closest",
        uirevision=uirev,
        clickmode="event+select",
        dragmode="pan",
    )

    fig.update_xaxes(visible=False, range=default_xrange)
    fig.update_yaxes(visible=False, range=default_yrange)

    if map_fit == "Original":
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig, label_indices


# ----------------------------
# Map (middle)
# ----------------------------
with col_mid:
    st.subheader("Map (click a dot)")

    fig, label_indices = build_map_figure()

    config = {
        "scrollZoom": True,
        "displaylogo": False,
        # makes WebGL dots stay sharp (reduces “blurry after selection” feeling)
        "plotGlPixelRatio": 2,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "resetScale2d"],
    }

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        config=config,
        on_select="rerun",
        selection_mode=["points"],
        key=PLOT_KEY,  # stable component id (prevents view reset)
    )

    # Handle clicks on either:
    #  - markers trace (curve 0 if no lines, but shifts if lines are on)
    #  - label trace (if user clicks text)
    clicked_genre: Optional[str] = None

    try:
        pts = event.selection.points  # type: ignore[attr-defined]
        if pts and isinstance(pts, list):
            p0 = pts[0]
            curve = p0.get("curve_number")
            pidx = p0.get("point_index")

            # Determine which trace user clicked:
            # Trace order: [edges... optional], markers always, labels optional
            # We'll find markers trace index dynamically: it's the first Scattergl trace in fig.data
            markers_trace_index = None
            labels_trace_index = None
            for i, tr in enumerate(fig.data):
                tname = tr.type
                if markers_trace_index is None and tname == "scattergl":
                    markers_trace_index = i
                # labels trace is the last "scatter" with mode text we added
                # (not robust by name, but reliable in this build)
                if tname == "scatter" and getattr(tr, "mode", "") == "text":
                    labels_trace_index = i

            if curve == markers_trace_index and isinstance(pidx, int):
                if 0 <= pidx < len(genres):
                    clicked_genre = genres[pidx]

            elif curve == labels_trace_index and isinstance(pidx, int):
                # map point index in label trace -> global df index
                if 0 <= pidx < len(label_indices):
                    gi = label_indices[pidx]
                    if 0 <= gi < len(genres):
                        clicked_genre = genres[gi]

    except Exception:
        clicked_genre = None

    if clicked_genre:
        st.session_state.selected_genre = clicked_genre
        st.session_state.active_path = []
        push_history(clicked_genre)
        st.rerun()

    st.caption("Tip: drag to pan; mouse wheel zooms. Use ‘Fit to screen’ if the map looks too thin.")


# ----------------------------
# Genre details (right)
# ----------------------------
with col_right:
    st.subheader("Genre details")

    sel = selected_genre
    if not sel:
        st.write("Pick a genre from the dropdown or click a dot on the map.")
    else:
        st.markdown(f"### {sel}")

        wiki_title = resolve_wiki_title_for_genre(sel)
        if not wiki_title:
            st.warning("No mapped Wikipedia page for this genre yet. (It may be listed in wiki_needs_review.csv.)")
        else:
            # Wikipedia intro
            try:
                info = wiki_intro_by_title(wiki_title)
                para = info.get("paragraph", "")
                url = info.get("url", "")
                if para:
                    st.write(para)
                else:
                    st.caption("No summary available for this page.")
                if url:
                    st.link_button("Open Wikipedia article", url)
            except Exception as e:
                st.warning(f"Could not load Wikipedia intro: {e}")

            st.markdown("---")

            # Spotify example
            token = spotify_get_token()
            market = st.session_state.market

            if not token:
                st.info("Spotify credentials not found. Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in Streamlit secrets.")
            else:
                example_track = None
                try:
                    example_track = pick_spotify_example_from_wikipedia(token, info.get("title", wiki_title), market=market)
                except Exception:
                    example_track = None

                if example_track and example_track.get("id"):
                    track_id = example_track["id"]
                    track_name = example_track.get("name", "Example track")
                    artist_names = ", ".join([a.get("name", "") for a in example_track.get("artists", []) if a.get("name")])

                    picked_from = example_track.get("_picked_from_artist")
                    if picked_from:
                        st.markdown(f"**Example (artist mentioned in intro):** {picked_from}")
                    else:
                        st.markdown("**Example:**")

                    st.write(f"{track_name} — {artist_names}")
                    st.components.v1.html(spotify_embed_html(track_id), height=170, scrolling=False)

                    open_url = example_track.get("external_urls", {}).get("spotify", "")
                    if open_url:
                        st.link_button("Open in Spotify", open_url)
                else:
                    st.caption("No Spotify example found for this genre right now.")


# Footer note
st.markdown(
    "<div class='small-muted'>Data: EveryNoise-derived genre attributes (genre, x, y, r, g, b). Connections are nearest-neighbor links.</div>",
    unsafe_allow_html=True,
)
