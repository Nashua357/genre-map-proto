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

/* Nicer sidebar spacing */
section[data-testid="stSidebar"] .block-container { padding-top: 1.0rem; }

/* Prevent long genre lists from blowing out layout */
.small-muted { font-size: 0.85rem; color: rgba(255,255,255,0.65); }
</style>
""",
    unsafe_allow_html=True,
)

UA = "genre-map-proto/1.0 (personal project)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API = "https://api.spotify.com/v1"

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
    # Streamlit Cloud secrets or environment vars
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
        params={"q": name, "type": "artist", "limit": 3, "market": market},
        timeout=30,
    )
    items = js.get("artists", {}).get("items", [])
    if not items:
        return None

    # Prefer exact-ish match
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
    # Use embed for a track (works without Web Playback SDK)
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

    # Normalize expected columns
    # Expected (from Every Noise–derived exports): genre, x, y, r, g, b
    colmap = {c.lower(): c for c in df.columns}

    def pick(name: str) -> Optional[str]:
        return colmap.get(name)

    genre_col = pick("genre") or pick("name")
    x_col = pick("x")
    y_col = pick("y")

    if not genre_col or not x_col or not y_col:
        raise ValueError("CSV must contain at least: genre, x, y columns.")

    df = df.rename(columns={genre_col: "genre", x_col: "x", y_col: "y"})

    # Colors: either r/g/b columns or a single hex column
    r_col, g_col, b_col = pick("r"), pick("g"), pick("b")
    if r_col and g_col and b_col:
        df = df.rename(columns={r_col: "r", g_col: "g", b_col: "b"})
    else:
        # If no rgb provided, create a pleasant default color
        df["r"], df["g"], df["b"] = 180, 180, 180

    # Ensure numeric
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["r"] = pd.to_numeric(df["r"], errors="coerce").fillna(180).astype(int)
    df["g"] = pd.to_numeric(df["g"], errors="coerce").fillna(180).astype(int)
    df["b"] = pd.to_numeric(df["b"], errors="coerce").fillna(180).astype(int)

    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    df["genre"] = df["genre"].astype(str)

    # Convenience: hex color for Plotly
    df["hex"] = df.apply(lambda row: f"rgb({int(row.r)},{int(row.g)},{int(row.b)})", axis=1)

    return df


@st.cache_data(show_spinner=False)
def load_wiki_title_maps() -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns (overrides_map, generated_map)
    Keys are normalized genre names; values are Wikipedia page titles.
    """
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
# Wikipedia fetch (by title)
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
    t = page.get("title")
    extract = (page.get("extract") or "").strip()
    paragraph = extract.split("\n")[0].strip() if extract else ""
    url = f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}" if t else ""
    return {"title": t or title, "paragraph": paragraph, "url": url}


@st.cache_data(show_spinner=False)
def wiki_lead_html(title: str) -> str:
    # Get HTML for first section (lead) so we can pull linked names (often artists/bands)
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
    """
    Very lightweight parsing: pull link titles from the first <p>...</p>.
    We don't need perfect HTML parsing here.
    """
    if not html:
        return []

    # First paragraph block
    m = re.search(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []

    p = m.group(1)

    # Extract title="..." or /wiki/...
    titles: List[str] = []
    for tm in re.finditer(r'title="([^"]+)"', p):
        t = tm.group(1)
        if t and t not in titles:
            titles.append(t)
        if len(titles) >= max_items:
            break

    # Fallback: /wiki/Name
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
    """
    Tries to choose an artist mentioned in the first paragraph.
    If that fails, falls back to a track search using the genre title.
    Returns a Spotify track object (dict) or None.
    """
    if not token:
        return None

    try:
        html = wiki_lead_html(wiki_title)
        link_titles = extract_first_paragraph_link_titles(html, max_items=25)
    except Exception:
        link_titles = []

    # Filter obvious non-artist links
    bad_words = [
        "music", "genre", "style", "history", "list", "album", "song", "record",
        "city", "state", "country", "united states", "england", "australia"
    ]
    filtered = []
    for t in link_titles:
        tl = norm_title(t)
        if any(bw in tl for bw in bad_words):
            continue
        # Skip very long titles (usually not artist names)
        if len(t) > 45:
            continue
        filtered.append(t)

    # Try top ~12 candidates for an artist match
    for name in filtered[:12]:
        try:
            artist = spotify_search_artist(token, name, market=market)
            if artist and artist.get("id"):
                top = spotify_top_track_for_artist(token, artist["id"], market=market)
                if top and top.get("id"):
                    # Attach artist name for display
                    top["_picked_from_artist"] = artist.get("name")
                    return top
        except Exception:
            continue

    # Fallback: just search a track with the wiki title
    try:
        tr = spotify_search_track(token, f"{wiki_title}", market=market)
        if tr and tr.get("id"):
            return tr
    except Exception:
        pass

    return None


# ----------------------------
# Nearest neighbors + path
# ----------------------------
@st.cache_data(show_spinner=False)
def compute_neighbors(xy: np.ndarray, k: int) -> List[List[int]]:
    """
    Returns neighbors list for each point.
    Tries sklearn if available, else uses a numpy chunk fallback.
    """
    n = xy.shape[0]
    k = int(clamp(k, 1, 25))

    # sklearn fast path
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto")
        nn.fit(xy)
        dists, idxs = nn.kneighbors(xy, return_distance=True)
        # drop self at position 0
        out = [list(row[1:]) for row in idxs]
        return out
    except Exception:
        pass

    # numpy fallback (chunked)
    xy_f = xy.astype(np.float32)
    out: List[List[int]] = []
    for i in range(n):
        d = np.sum((xy_f - xy_f[i]) ** 2, axis=1)
        # partial sort
        idx = np.argpartition(d, min(k + 1, n - 1))[: min(k + 1, n)]
        idx = idx[idx != i]
        # sort those by distance
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

    # Reconstruct
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# ----------------------------
# Label selection (so labels are readable)
# ----------------------------
@st.cache_data(show_spinner=False)
def pick_label_points(df: pd.DataFrame, max_labels: int = 220) -> np.ndarray:
    """
    Pick a spread-out set of points to label in the default view.
    (Grid sampling so labels don't all pile up.)
    """
    max_labels = int(clamp(max_labels, 50, 600))
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    # Grid bins
    bins = int(np.sqrt(max_labels) * 1.6)
    bins = int(clamp(bins, 10, 50))

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
    st.session_state.selected_genre = ""  # empty means none
if "destination_genre" not in st.session_state:
    st.session_state.destination_genre = ""
if "selection_history" not in st.session_state:
    st.session_state.selection_history = []  # list[str]
if "active_path" not in st.session_state:
    st.session_state.active_path = []  # list[int]
if "last_clicked_idx" not in st.session_state:
    st.session_state.last_clicked_idx = None
if "market" not in st.session_state:
    st.session_state.market = "AU"

UIREV = "genre-map-uirev-1"  # constant to preserve pan/zoom across reruns


def push_history(genre: str, max_items: int = 12) -> None:
    g = (genre or "").strip()
    if not g:
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
    st.markdown("## Spotify examples")
    market = st.selectbox(
        "Country/market",
        ["AU", "US", "GB", "CA", "NZ", "DE", "FR", "BR", "JP"],
        index=["AU", "US", "GB", "CA", "NZ", "DE", "FR", "BR", "JP"].index(st.session_state.market),
    )
    st.session_state.market = market

    st.markdown("---")
    st.markdown("## View")
    map_fit = st.radio("Map shape", ["Fit to screen", "Original"], index=0)


# ----------------------------
# Load data
# ----------------------------
try:
    if data_source == "Upload CSV" and uploaded is not None:
        df = pd.read_csv(uploaded)
        # Normalize similarly to repo loader
        colmap = {c.lower(): c for c in df.columns}
        if "genre" not in colmap or "x" not in colmap or "y" not in colmap:
            st.error("Uploaded CSV must include columns: genre, x, y (and optionally r,g,b).")
            st.stop()
        df = df.rename(columns={colmap["genre"]: "genre", colmap["x"]: "x", colmap["y"]: "y"})
        if all(c in colmap for c in ["r", "g", "b"]):
            df = df.rename(columns={colmap["r"]: "r", colmap["g"]: "g", colmap["b"]: "b"})
        else:
            df["r"], df["g"], df["b"] = 180, 180, 180
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
        df["hex"] = df.apply(lambda row: f"rgb({int(row.r)},{int(row.g)},{int(row.b)})", axis=1)
    else:
        df = load_genre_data_from_repo()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

genres = df["genre"].tolist()
genre_to_idx = {g: i for i, g in enumerate(genres)}

xy = df[["x", "y"]].to_numpy(dtype=np.float32)
neighbors = compute_neighbors(xy, k=k)
adj = build_adjacency(neighbors)

label_idxs_default = pick_label_points(df, max_labels=240)

# ----------------------------
# Main layout
# ----------------------------
st.title("Phase 1 Prototype — Genre Map")

col_left, col_mid, col_right = st.columns([0.33, 0.42, 0.25], gap="large")

# ----------------------------
# Controls (left main panel)
# ----------------------------
with col_left:
    st.subheader("Controls")

    search = st.text_input("Search genre", value="")

    # Filter select options (but keep full list for stable indices)
    if search.strip():
        filtered = [g for g in genres if search.lower() in g.lower()]
        if not filtered:
            filtered = genres
    else:
        filtered = genres

    options = ["(none)"] + filtered
    # Keep widget key stable; do NOT set default + session state in a conflicting way
    current = st.session_state.selected_genre
    if current and current not in filtered and current in genres:
        # ensure selected stays selectable
        options = ["(none)"] + [current] + filtered

    selected_label = current if current else "(none)"
    sel = st.selectbox("Selected genre", options, index=options.index(selected_label), key="selected_genre_dropdown")
    selected_genre = "" if sel == "(none)" else sel

    if selected_genre != st.session_state.selected_genre:
        st.session_state.selected_genre = selected_genre
        st.session_state.active_path = []
        if selected_genre:
            push_history(selected_genre)

    # Show closest genres
    if st.session_state.selected_genre and st.session_state.selected_genre in genre_to_idx:
        si = genre_to_idx[st.session_state.selected_genre]
        close = [genres[j] for j in neighbors[si]]
        st.markdown("**Closest genres**")
        st.write(", ".join(close))

    # Path finder
    st.subheader("Path finder")
    if enable_path and st.session_state.selected_genre and st.session_state.selected_genre in genre_to_idx:
        st.text_input("Search destination", value="", key="dest_search")
        dest_search = st.session_state.get("dest_search", "").strip().lower()
        dest_filtered = [g for g in genres if dest_search in g.lower()] if dest_search else genres

        dest_options = ["(none)"] + dest_filtered
        current_dest = st.session_state.destination_genre
        dest_label = current_dest if current_dest else "(none)"
        if dest_label not in dest_options and current_dest in genres:
            dest_options = ["(none)"] + [current_dest] + dest_filtered

        dest_sel = st.selectbox("Destination genre", dest_options, index=dest_options.index(dest_label), key="dest_genre_dropdown")
        destination = "" if dest_sel == "(none)" else dest_sel
        st.session_state.destination_genre = destination

        if st.button("Find shortest path", use_container_width=True):
            start_idx = genre_to_idx[st.session_state.selected_genre]
            if destination and destination in genre_to_idx:
                end_idx = genre_to_idx[destination]
                path = shortest_path(adj, start_idx, end_idx)
                st.session_state.active_path = path
                if path:
                    st.success(f"Path found: {len(path)} steps.")
                else:
                    st.warning("No path found (graph may be disconnected at this k).")
            else:
                st.warning("Pick a destination genre first.")
    else:
        st.caption("Enable Path finder and select a genre to use this.")


# ----------------------------
# Build Plotly figure (middle)
# ----------------------------
def build_edges_trace(edge_pairs: List[Tuple[int, int]], color: str, width: float, opacity: float) -> go.Scattergl:
    xs = []
    ys = []
    for a, b in edge_pairs:
        xs.extend([df.at[a, "x"], df.at[b, "x"], None])
        ys.extend([df.at[a, "y"], df.at[b, "y"], None])
    return go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity,
        hoverinfo="skip",
        showlegend=False,
    )


def build_map_figure() -> go.Figure:
    n = len(df)

    selected = st.session_state.selected_genre
    selected_idx = genre_to_idx.get(selected) if selected else None
    path_idxs = st.session_state.active_path or []

    # Determine highlighting sets
    highlight_nodes = set()
    neighbor_nodes = set()

    if selected_idx is not None:
        highlight_nodes.add(selected_idx)
        for j in neighbors[selected_idx]:
            neighbor_nodes.add(j)

    for p in path_idxs:
        highlight_nodes.add(p)

    # Marker styling
    base_opacity = 0.18 if (selected_idx is not None or path_idxs) else 0.95
    base_size = 5.5 if (selected_idx is not None or path_idxs) else 6.0

    # Colors
    colors = df["hex"].tolist()

    # Base markers
    marker_opacity = np.full(n, base_opacity, dtype=np.float32)
    marker_size = np.full(n, base_size, dtype=np.float32)

    # Make everything visible in default view
    if selected_idx is None and not path_idxs:
        marker_opacity[:] = 0.95
        marker_size[:] = 6.0

    # Highlight selected + neighbors + path
    for idx in neighbor_nodes:
        marker_opacity[idx] = 0.70
        marker_size[idx] = 7.2

    for idx in highlight_nodes:
        marker_opacity[idx] = 0.98
        marker_size[idx] = 9.0

    # Labels:
    label_idxs = set()
    if selected_idx is None and not path_idxs:
        label_idxs.update(label_idxs_default.tolist())
    else:
        if selected_idx is not None:
            label_idxs.add(selected_idx)
            label_idxs.update(list(neighbor_nodes)[:10])
        label_idxs.update(path_idxs)

    # Build traces
    fig = go.Figure()

    # Edges
    if show_lines:
        # default edges (faint)
        if selected_idx is None and not path_idxs:
            edges = []
            seen = set()
            for i in range(n):
                for j in neighbors[i]:
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) not in seen:
                        seen.add((a, b))
                        edges.append((a, b))
            fig.add_trace(build_edges_trace(edges, color="rgba(120,150,255,0.35)", width=0.8, opacity=0.6))
        else:
            # Show local edges around selected + path (clear), keep others hidden
            edges_local = []
            seen = set()

            focus = set()
            if selected_idx is not None:
                focus.add(selected_idx)
                focus.update(neighbor_nodes)
            focus.update(path_idxs)

            for i in focus:
                for j in neighbors[i]:
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) not in seen:
                        seen.add((a, b))
                        edges_local.append((a, b))

            if edges_local:
                fig.add_trace(build_edges_trace(edges_local, color="rgba(120,150,255,0.55)", width=1.1, opacity=0.85))

            # Path edges thicker
            if len(path_idxs) >= 2:
                path_edges = list(zip(path_idxs[:-1], path_idxs[1:]))
                fig.add_trace(build_edges_trace(path_edges, color="rgba(255,255,255,0.85)", width=2.6, opacity=0.95))

    # Markers
    fig.add_trace(
        go.Scattergl(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=colors,
                opacity=marker_opacity,
                line=dict(width=0),
            ),
            hovertext=df["genre"],
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Labels trace (separate so it stays readable)
    if label_idxs:
        li = sorted(label_idxs)
        fig.add_trace(
            go.Scattergl(
                x=df.loc[li, "x"],
                y=df.loc[li, "y"],
                mode="text",
                text=df.loc[li, "genre"],
                textposition="top center",
                textfont=dict(size=10, color="rgba(255,255,255,0.85)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Layout + preserve view
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_dark",
        height=620,
        uirevision=UIREV,  # keeps zoom/pan stable across reruns
        hovermode="closest",
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)

    return fig


with col_mid:
    st.subheader("Map (click a dot)")
    fig = build_map_figure()

    # Streamlit selection event (works for point clicks when clickmode includes select)
    fig.update_layout(clickmode="event+select", dragmode="pan")

    config = {
        "scrollZoom": True,  # mouse-wheel zoom
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "autoScale2d",
            "resetScale2d",
        ],
    }

    # IMPORTANT: on_select="rerun" lets Streamlit capture point selection
    event = st.plotly_chart(fig, use_container_width=True, config=config, on_select="rerun", selection_mode=["points"])

    clicked_idx = None
    try:
        # Streamlit selection payload
        pts = event.selection.points  # type: ignore[attr-defined]
        if pts and isinstance(pts, list):
            # plotly index is under "point_index"
            clicked_idx = pts[0].get("point_index", None)
    except Exception:
        clicked_idx = None

    if clicked_idx is not None and isinstance(clicked_idx, int):
        if 0 <= clicked_idx < len(genres):
            g = genres[clicked_idx]
            st.session_state.selected_genre = g
            st.session_state.active_path = []
            push_history(g)
            st.session_state.last_clicked_idx = clicked_idx
            st.rerun()

    st.caption("Tip: drag to pan; mouse wheel zooms. Click a dot to load details.")


# ----------------------------
# Genre details (right)
# ----------------------------
with col_right:
    st.subheader("Genre details")

    selected = st.session_state.selected_genre
    if not selected:
        st.write("Pick a genre from the dropdown or click a dot on the map.")
        st.stop()

    st.markdown(f"### {selected}")

    wiki_title = resolve_wiki_title_for_genre(selected)
    if not wiki_title:
        st.info("No mapped Wikipedia page for this genre yet. (It may be in needs_review.)")
        st.stop()

    # Wikipedia intro
    try:
        info = wiki_intro_by_title(wiki_title)
    except Exception as e:
        st.warning(f"Could not load Wikipedia intro: {e}")
        st.stop()

    para = info.get("paragraph", "")
    url = info.get("url", "")

    if para:
        st.write(para)
    else:
        st.caption("No summary available.")

    if url:
        st.link_button("Open Wikipedia article", url)

    st.markdown("---")

    # Spotify example based on an artist mentioned in the first paragraph (best effort)
    token = spotify_get_token()
    market = st.session_state.market

    example_track = None
    example_title = ""
    try:
        example_track = pick_spotify_example_from_wikipedia(token, info.get("title", wiki_title), market=market)
    except Exception:
        example_track = None

    if not token:
        st.info("Spotify credentials not found. Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in Streamlit secrets.")
        st.stop()

    if example_track and example_track.get("id"):
        track_id = example_track["id"]
        track_name = example_track.get("name", "Example track")
        artist_names = ", ".join([a.get("name", "") for a in example_track.get("artists", []) if a.get("name")])

        picked_from = example_track.get("_picked_from_artist")
        if picked_from:
            example_title = f"Example (artist mentioned in intro): {picked_from}"
        else:
            example_title = "Example"

        st.markdown(f"**{example_title}:** {track_name} — {artist_names}")

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
