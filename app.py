import base64
import heapq
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
from streamlit_plotly_events import plotly_events

# =========================
# URLs
# =========================
DATA_URL = "https://raw.githubusercontent.com/AyrtonB/EveryNoise-Watch/main/data/genre_attrs.csv"

# Wikipedia MediaWiki API (more reliable than REST summary for our use)
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_UA = "genre-map-proto/1.0 (personal project)"

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"


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
            h = df[hex_col].astype(str).str.strip()
            h = h.apply(lambda s: ("#" + s.lstrip("#")) if len(s.lstrip("#")) == 6 else "#888888")
            df["hex_colour"] = h

    df = df.reset_index(drop=True)
    return df


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
# Wikipedia
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_wikipedia_intro(genre_name: str) -> Dict[str, Optional[str]]:
    """
    Robust: search -> fetch intro extract + canonical url (all via MediaWiki API).
    Returns {title, paragraph, url}.
    """
    headers = {"User-Agent": WIKI_UA}

    # 1) Search for best page
    queries = [
        f"{genre_name} music genre",
        f"{genre_name} (music)",
        genre_name,
    ]

    title = None
    for q in queries:
        try:
            r = requests.get(
                WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": q,
                    "srlimit": 1,
                    "format": "json",
                },
                headers=headers,
                timeout=15,
            )
            r.raise_for_status()
            js = r.json()
            hits = js.get("query", {}).get("search", [])
            if hits:
                title = hits[0].get("title")
                if title:
                    break
        except Exception:
            continue

    if not title:
        return {"title": None, "paragraph": None, "url": None}

    # 2) Fetch first paragraph + full URL
    try:
        r2 = requests.get(
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
        r2.raise_for_status()
        js2 = r2.json()
        pages = js2.get("query", {}).get("pages", {})
        page = next(iter(pages.values())) if pages else {}
        extract = (page.get("extract") or "").strip()
        paragraph = extract.split("\n")[0].strip() if extract else None
        url = page.get("fullurl")
        return {"title": title, "paragraph": paragraph, "url": url}
    except Exception:
        return {"title": title, "paragraph": None, "url": None}


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

    # 1) Track search with genre:"..."
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

    # 2) Playlist fallback
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
    hist = st.session_state.setdefault("genre_history", [])
    if genre and (len(hist) == 0 or hist[0] != genre):
        hist = [genre] + [g for g in hist if g != genre]
        st.session_state["genre_history"] = hist[:max_len]


# =========================
# Plotly figure
# =========================
def make_figure(
    df_plot: pd.DataFrame,
    selected_idx: int,
    neighbor_idxs: Set[int],
    edges: List[Tuple[int, int, float]],
    show_edges: bool,
    path: Optional[List[int]],
) -> go.Figure:
    n = len(df_plot)

    size = np.full(n, 7.0)
    opacity = np.full(n, 0.80)

    for i in neighbor_idxs:
        size[i] = 11.0
        opacity[i] = 1.0

    size[selected_idx] = 16.0
    opacity[selected_idx] = 1.0

    if path:
        for i in path:
            size[i] = max(size[i], 12.0)
            opacity[i] = 1.0

    fig = go.Figure()

    if show_edges and edges:
        xs, ys = [], []
        for u, v, _d in edges:
            xs.extend([df_plot.at[u, "x"], df_plot.at[v, "x"], None])
            ys.extend([df_plot.at[u, "y"], df_plot.at[v, "y"], None])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=1),
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if path and len(path) >= 2:
        px, py = [], []
        for a, b in zip(path[:-1], path[1:]):
            px.extend([df_plot.at[a, "x"], df_plot.at[b, "x"], None])
            py.extend([df_plot.at[a, "y"], df_plot.at[b, "y"], None])
        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="lines",
                line=dict(width=3),
                opacity=0.85,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df_plot["x"],
            y=df_plot["y"],
            mode="markers",
            marker=dict(
                size=size,
                color=df_plot["hex_colour"].tolist(),
                opacity=opacity,
                symbol="circle",
            ),
            text=df_plot["genre"],
            hovertemplate="<b>%{text}</b><extra></extra>",
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

pending = st.session_state.pop("pending_genre", None)
if pending:
    st.session_state["selected_genre_dropdown"] = pending

with st.sidebar:
    st.header("Selection history")
    hist = st.session_state.get("genre_history", [])
    if not hist:
        st.caption("Click dots or pick genres to build history.")
    else:
        for i, g in enumerate(hist[:12]):
            if st.button(g, key=f"hist_{i}", use_container_width=True):
                st.session_state["selected_genre_dropdown"] = g
                st.session_state["pending_genre"] = g
                st.rerun()
        cols = st.columns(2)
        with cols[0]:
            if st.button("Clear", use_container_width=True):
                st.session_state["genre_history"] = []
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
    st.header("Spotify examples")
    market = st.selectbox("Country/market", ["AU", "US", "GB", "CA", "NZ", "DE", "FR"], index=0)

    st.divider()
    st.header("View")
    view_mode = st.radio("Map shape", ["Fit to screen", "Original"], index=0)

df = load_genre_data("upload" if source == "Upload CSV" else "web", uploaded_bytes)

if "selected_genre_dropdown" not in st.session_state:
    st.session_state["selected_genre_dropdown"] = df["genre"].iloc[0] if len(df) else ""

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

with col_controls:
    st.subheader("Controls")

    q = st.text_input("Search genre", value="", key="start_query")
    if q.strip():
        mask = df["genre"].str.contains(q.strip(), case=False, na=False)
        candidates = df.loc[mask, "genre"].tolist()
    else:
        candidates = df["genre"].head(800).tolist()

    if not candidates:
        st.warning("No matches.")
        st.stop()

    cur = st.session_state["selected_genre_dropdown"]
    if cur and cur not in candidates and cur in df["genre"].values:
        candidates = [cur] + candidates

    chosen = st.selectbox("Selected genre", candidates, index=0, key="selected_genre_dropdown")
    push_history(chosen)

    selected_idx = int(df.index[df["genre"] == chosen][0])
    neighbor_idxs = {v for v, _d in adj[selected_idx]}
    neighbor_list = df.loc[list(neighbor_idxs), "genre"].sort_values().tolist()

    st.caption("Closest genres")
    st.write(", ".join(neighbor_list[:25]) + (" ..." if len(neighbor_list) > 25 else ""))

    path: List[int] = []
    if st.session_state.get("enable_path", True):
        st.markdown("### Path finder")
        dest_q = st.text_input("Search destination", value="", key="dest_query")

        if dest_q.strip():
            dest_mask = df["genre"].str.contains(dest_q.strip(), case=False, na=False)
            end_candidates = df.loc[dest_mask, "genre"].tolist()
        else:
            end_candidates = stable_sample_genres(df["genre"], n=500)

        if end_candidates:
            end = st.selectbox("Destination genre", end_candidates, index=0, key="dest_genre")
            st.markdown('<div style="height: 90px;"></div>', unsafe_allow_html=True)

            if st.button("Find shortest path"):
                end_idx = int(df.index[df["genre"] == end][0])
                path = dijkstra_path(adj, selected_idx, end_idx)
                if path:
                    st.success(f"Path found: {len(path)} steps.")
                else:
                    st.error("No path found. Try increasing connections.")

with col_map:
    st.subheader("Map (click a dot)")

    show_edges = st.session_state.get("show_edges", True)

    if path and len(path) >= 2:
        edges_to_draw = []
    else:
        focus = {selected_idx} | neighbor_idxs
        edges_to_draw = [(u, v, d) for (u, v, d) in undirected_edges if u in focus and v in focus]

    fig = make_figure(
        df_plot=df_plot,
        selected_idx=selected_idx,
        neighbor_idxs=neighbor_idxs,
        edges=edges_to_draw,
        show_edges=show_edges,
        path=path if path else None,
    )

    clicked = plotly_events(
        fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        override_height=720,
        key="plotly_click",
    )

    if clicked:
        pi = clicked[0].get("pointIndex")
        if pi is not None and 0 <= int(pi) < len(df):
            clicked_genre = df.loc[int(pi), "genre"]
            st.session_state["pending_genre"] = clicked_genre
            push_history(clicked_genre)
            st.rerun()

with col_details:
    st.subheader("Genre details")
    genre_name = st.session_state.get("selected_genre_dropdown", "")

    if not genre_name:
        st.info("Select a genre to see details.")
        st.stop()

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

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Pin to player", use_container_width=True):
                    st.session_state["pinned_type"] = ex["type"]
                    st.session_state["pinned_id"] = ex["id"]
                    st.session_state["pinned_url"] = ex.get("url")
                    st.rerun()
            with c2:
                if ex.get("url"):
                    st.markdown(f"[Open in Spotify]({ex['url']})")
        else:
            st.caption("Couldn’t find a Spotify example for this exact genre name.")
            st.markdown(f"[Search this in Spotify](https://open.spotify.com/search/{quote(genre_name)})")

    st.divider()

    pinned_type = st.session_state.get("pinned_type")
    pinned_id = st.session_state.get("pinned_id")
    pinned_url = st.session_state.get("pinned_url")

    st.markdown("### Pinned player")
    if pinned_type and pinned_id:
        st.components.v1.html(spotify_embed_html(pinned_type, pinned_id, height=152), height=170, scrolling=False)
        if pinned_url:
            st.markdown(f"[Open pinned item in Spotify]({pinned_url})")
        if st.button("Clear pinned player", use_container_width=True):
            st.session_state.pop("pinned_type", None)
            st.session_state.pop("pinned_id", None)
            st.session_state.pop("pinned_url", None)
            st.rerun()
    else:
        st.caption("Click **Pin to player** on an example to keep it playing while you browse other genres.")
