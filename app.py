import base64
import heapq
import time
import zlib
from typing import Optional, List, Tuple, Set, Dict
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.neighbors import NearestNeighbors

# EveryNoise-derived genre attributes (genre,x,y,hex_colour) mirrored in a public repo
DATA_URL = "https://raw.githubusercontent.com/AyrtonB/EveryNoise-Watch/main/data/genre_attrs.csv"

WIKI_UA = "genre-map-proto/1.0 (personal project)"
WIKI_OPENSEARCH = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"


# ---------------------------
# Helpers: data loading
# ---------------------------
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

    # Color: support either r/g/b or a hex color column like hex_colour
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
            # normalize to "#rrggbb"
            h = h.apply(lambda s: ("#" + s.lstrip("#")) if len(s.lstrip("#")) == 6 else "#888888")
            df["hex_colour"] = h

        # also create r/g/b for compatibility
        hh = df["hex_colour"].str.lstrip("#")
        df["r"] = hh.str[0:2].apply(lambda s: int(s, 16) if len(s) == 2 else 136)
        df["g"] = hh.str[2:4].apply(lambda s: int(s, 16) if len(s) == 2 else 136)
        df["b"] = hh.str[4:6].apply(lambda s: int(s, 16) if len(s) == 2 else 136)

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def stable_sample_genres(genres: pd.Series, n: int = 400) -> List[str]:
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


# ---------------------------
# Helpers: Wikipedia
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_wikipedia_first_paragraph(genre_name: str) -> Dict[str, Optional[str]]:
    """
    Returns {title, paragraph, url} using Wikipedia search -> summary.
    """
    # Search for best page
    search_term = f"{genre_name} music genre"
    try:
        r = requests.get(
            WIKI_OPENSEARCH,
            params={
                "action": "opensearch",
                "search": search_term,
                "limit": 1,
                "namespace": 0,
                "format": "json",
            },
            headers={"User-Agent": WIKI_UA},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        titles = data[1] if len(data) > 1 else []
        if not titles:
            return {"title": None, "paragraph": None, "url": None}
        title = titles[0]
    except Exception:
        return {"title": None, "paragraph": None, "url": None}

    # Pull summary paragraph
    try:
        r2 = requests.get(
            f"{WIKI_SUMMARY}{quote(title)}",
            headers={"User-Agent": WIKI_UA},
            timeout=15,
        )
        r2.raise_for_status()
        js = r2.json()

        extract = js.get("extract") or ""
        paragraph = extract.split("\n")[0].strip() if isinstance(extract, str) else None

        url = (
            js.get("content_urls", {})
            .get("desktop", {})
            .get("page")
        )

        return {"title": title, "paragraph": paragraph, "url": url}
    except Exception:
        return {"title": title, "paragraph": None, "url": None}


# ---------------------------
# Helpers: Spotify Web API (Client Credentials)
# ---------------------------
def _get_spotify_creds() -> Tuple[Optional[str], Optional[str]]:
    try:
        return st.secrets.get("SPOTIFY_CLIENT_ID"), st.secrets.get("SPOTIFY_CLIENT_SECRET")
    except Exception:
        return None, None


def get_spotify_token() -> Optional[str]:
    # reuse until near expiry
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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def spotify_example_for_genre(genre_name: str, market: str = "AU") -> Optional[Dict[str, str]]:
    """
    Returns one example item to play for a genre.
    Tries a track using the genre filter; falls back to a playlist search.
    Dict: {type, id, name, subtitle, url}
    """
    tok = get_spotify_token()
    if not tok:
        return None

    headers = {"Authorization": f"Bearer {tok}"}

    # 1) Try track search with genre:"..."
    try:
        r = requests.get(
            SPOTIFY_SEARCH_URL,
            params={
                "q": f'genre:"{genre_name}"',
                "type": "track",
                "limit": 1,
                "market": market,
            },
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("tracks", {}).get("items", [])
        if items:
            t = items[0]
            return {
                "type": "track",
                "id": t["id"],
                "name": t["name"],
                "subtitle": t["artists"][0]["name"] if t.get("artists") else "",
                "url": t.get("external_urls", {}).get("spotify", ""),
            }
    except Exception:
        pass

    # 2) Fallback: playlist search by text
    try:
        r = requests.get(
            SPOTIFY_SEARCH_URL,
            params={
                "q": genre_name,
                "type": "playlist",
                "limit": 1,
                "market": market,
            },
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("playlists", {}).get("items", [])
        if items:
            p = items[0]
            return {
                "type": "playlist",
                "id": p["id"],
                "name": p["name"],
                "subtitle": "Playlist",
                "url": p.get("external_urls", {}).get("spotify", ""),
            }
    except Exception:
        pass

    return None


def spotify_embed_html(item_type: str, item_id: str) -> str:
    # supports track/playlist embeds
    if item_type not in {"track", "playlist"}:
        item_type = "track"
    src = f"https://open.spotify.com/embed/{item_type}/{item_id}"
    return f"""
      <iframe style="border-radius:12px"
        src="{src}"
        width="100%" height="152" frameborder="0"
        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
        loading="lazy"></iframe>
    """


# ---------------------------
# Dialog popup
# ---------------------------
def _dialog(title: str):
    if hasattr(st, "dialog"):
        return st.dialog(title)
    return st.experimental_dialog(title)


@_dialog("Genre details")
def show_genre_dialog(genre_name: str, market: str):
    st.subheader(genre_name)

    # Wikipedia paragraph
    wiki = fetch_wikipedia_first_paragraph(genre_name)
    if wiki.get("paragraph"):
        st.write(wiki["paragraph"])
    else:
        st.info("No clear Wikipedia summary found for this genre name.")

    if wiki.get("url"):
        st.markdown(f"[Open Wikipedia article]({wiki['url']})")

    st.divider()

    # Spotify example
    ex = spotify_example_for_genre(genre_name, market=market)
    if not ex:
        st.warning("Couldn’t find a Spotify example for this genre.")
        st.caption("Some niche genres don’t map cleanly to Spotify’s genre filter/search.")
        return

    st.markdown(f"**Example:** {ex['name']} — {ex['subtitle']}")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Load into player", use_container_width=True):
            st.session_state["player_type"] = ex["type"]
            st.session_state["player_id"] = ex["id"]
            st.session_state["player_url"] = ex.get("url")
            st.rerun()
    with col2:
        if ex.get("url"):
            st.markdown(f"[Open in Spotify]({ex['url']})")
    with col3:
        st.caption("Tip: press play in the corner player, then close this popup — it keeps playing.")

    # Optional: show a preview embed inside the popup too
    with st.expander("Preview in this popup"):
        st.components.v1.html(spotify_embed_html(ex["type"], ex["id"]), height=170, scrolling=False)


# ---------------------------
# Figure
# ---------------------------
def make_figure(
    df_plot: pd.DataFrame,
    selected_idx: int,
    neighbor_idxs: Set[int],
    edges_to_draw: List[Tuple[int, int, float]],
    path: Optional[List[int]],
) -> go.Figure:
    n = len(df_plot)
    size = np.full(n, 5.0)
    opacity = np.full(n, 0.22)

    for i in neighbor_idxs:
        size[i] = 9.0
        opacity[i] = 0.9

    size[selected_idx] = 14.0
    opacity[selected_idx] = 1.0

    path_set = set(path or [])
    for i in path_set:
        size[i] = max(size[i], 11.0)
        opacity[i] = 1.0

    fig = go.Figure()

    # edges
    if edges_to_draw:
        xs, ys = [], []
        for u, v, _d in edges_to_draw:
            xs.extend([df_plot.at[u, "x"], df_plot.at[v, "x"], None])
            ys.extend([df_plot.at[u, "y"], df_plot.at[v, "y"], None])

        fig.add_trace(
            go.Scattergl(
                x=xs, y=ys,
                mode="lines",
                line=dict(width=1),
                opacity=0.30,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # points
    fig.add_trace(
        go.Scattergl(
            x=df_plot["x"],
            y=df_plot["y"],
            mode="markers",
            marker=dict(size=size, color=df_plot["hex_colour"], opacity=opacity),
            customdata=np.arange(n),
            hovertemplate="<b>%{text}</b><extra></extra>",
            text=df_plot["genre"],
            showlegend=False,
        )
    )

    fig.update_layout(
        height=720,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        dragmode="pan",
    )
    return fig


# ---------------------------
# App UI
# ---------------------------
st.markdown(
    """
<style>
/* pointer cursor for dropdowns */
div[data-baseweb="select"] * { cursor: pointer !important; }
div[data-baseweb="select"] input { caret-color: transparent; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Phase 1 Prototype — Genre Map")

with st.sidebar:
    st.header("Data")
    source = st.radio("Load dataset from:", ["Web (recommended)", "Upload CSV"], index=0)
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload genre CSV", type=["csv"])
        if not uploaded:
            st.stop()
        df = load_genre_data("upload", uploaded.getvalue())
    else:
        df = load_genre_data("web")

    st.divider()
    st.header("Connections")
    k = st.slider("Connections per genre", 2, 20, 8)
    show_pathfinder = st.checkbox("Enable path finder", value=True)

    st.divider()
    st.header("Spotify examples")
    market = st.selectbox("Country/market", ["AU", "US", "GB", "CA", "NZ", "DE", "FR"], index=0)

    st.divider()
    st.header("View")
    view_mode = st.radio("Map shape", ["Fit to screen", "Original"], index=0)

    st.divider()
    st.header("Now Playing")
    player_type = st.session_state.get("player_type")
    player_id = st.session_state.get("player_id")
    player_url = st.session_state.get("player_url")

    if player_type and player_id:
        st.components.v1.html(spotify_embed_html(player_type, player_id), height=170, scrolling=False, key="player_embed")
        if player_url:
            st.markdown(f"[Open in Spotify]({player_url})")
        if st.button("Clear player"):
            st.session_state.pop("player_type", None)
            st.session_state.pop("player_id", None)
            st.session_state.pop("player_url", None)
            st.rerun()
    else:
        st.caption("Load a genre example to play it here.")


# Main map geometry
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

# Single source of truth for selection
if "selected_genre" not in st.session_state:
    st.session_state["selected_genre"] = df["genre"].iloc[0] if len(df) else ""

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Controls")

    # Put dropdowns high so they are more likely to open downward
    q = st.text_input("Search start genre", value="", key="start_query")

    if q.strip():
        mask = df["genre"].str.contains(q.strip(), case=False, na=False)
        candidates = df.loc[mask, "genre"].tolist()
    else:
        candidates = df["genre"].head(500).tolist()

    if not candidates:
        st.warning("No matches.")
        st.stop()

    # Ensure current selection stays in the list
    cur = st.session_state["selected_genre"]
    if cur and cur not in candidates and cur in df["genre"].values:
        candidates = [cur] + candidates

    selected = st.selectbox("Selected genre", candidates, index=0, key="selected_genre")

    # Button to open details without clicking the map
    if st.button("Show details popup"):
        st.session_state["open_genre_dialog"] = selected
        st.rerun()

    selected_idx = int(df.index[df["genre"] == selected][0])
    neighbor_idxs = {v for v, _d in adj[selected_idx]}
    neighbor_list = df.loc[list(neighbor_idxs), "genre"].sort_values().tolist()

    st.caption("Closest genres")
    st.write(", ".join(neighbor_list[:25]) + (" ..." if len(neighbor_list) > 25 else ""))

    path: List[int] = []
    path_edges: List[Tuple[int, int, float]] = []

    if show_pathfinder:
        st.markdown("### Path finder")

        dest_q = st.text_input("Search destination", value="", key="dest_query")
        if dest_q.strip():
            dest_mask = df["genre"].str.contains(dest_q.strip(), case=False, na=False)
            end_candidates = df.loc[dest_mask, "genre"].tolist()
        else:
            end_candidates = stable_sample_genres(df["genre"], n=500)

        if end_candidates:
            # keep current dest stable
            cur_dest = st.session_state.get("dest_genre")
            if cur_dest and cur_dest not in end_candidates and cur_dest in df["genre"].values:
                end_candidates = [cur_dest] + end_candidates

            end = st.selectbox("Destination genre", end_candidates, index=0, key="dest_genre")

            # Spacer encourages dropdown to open downward
            st.markdown('<div style="height: 120px;"></div>', unsafe_allow_html=True)

            if st.button("Find shortest path"):
                end_idx = int(df.index[df["genre"] == end][0])
                path = dijkstra_path(adj, selected_idx, end_idx)
                if len(path) >= 2:
                    path_edges = [(a, b, 0.0) for a, b in zip(path[:-1], path[1:])]
                    st.success(f"Path found: {len(path)} steps.")
                elif path:
                    st.info("Start and end are the same genre.")
                else:
                    st.error("No path found. Try increasing connections.")


with right:
    st.subheader("Map (click a genre dot for details)")

    # edges to draw: focused neighborhood unless a path exists
    if path_edges:
        edges_to_draw = path_edges
    else:
        focus = {selected_idx} | neighbor_idxs
        edges_to_draw = [(u, v, d) for (u, v, d) in undirected_edges if u in focus and v in focus]

    fig = make_figure(
        df_plot=df_plot,
        selected_idx=selected_idx,
        neighbor_idxs=neighbor_idxs,
        edges_to_draw=edges_to_draw,
        path=path if path else None,
    )

    # Click handling
    evt = st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True},
        on_select="rerun",
        selection_mode=("points",),
        key="genre_map",
    )

    # Read clicked point
    clicked_genre = None
    try:
        if evt and getattr(evt, "selection", None):
            sel = evt.selection
            points = sel.get("points", []) if isinstance(sel, dict) else []
            if points:
                p0 = points[0]
                idx = p0.get("point_index")
                if idx is not None and 0 <= int(idx) < len(df):
                    clicked_genre = df.loc[int(idx), "genre"]
    except Exception:
        clicked_genre = None

    # If user clicked a different genre, update selection and open popup
    if clicked_genre:
        last = st.session_state.get("last_clicked_genre")
        if clicked_genre != last:
            st.session_state["last_clicked_genre"] = clicked_genre
            st.session_state["selected_genre"] = clicked_genre
            st.session_state["open_genre_dialog"] = clicked_genre
            st.rerun()


# Open dialog if requested
to_open = st.session_state.pop("open_genre_dialog", None)
if to_open:
    show_genre_dialog(to_open, market=market)

st.caption(
    "Tip: play something in the Now Playing corner, then close the popup — it keeps playing."
)
