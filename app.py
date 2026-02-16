import heapq
from typing import Optional, List, Tuple, Set

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors

# EveryNoise-derived genre attributes mirrored in a public repo.
# NOTE: This file currently uses columns: genre, x, y, hex_colour
DATA_URL = "https://raw.githubusercontent.com/AyrtonB/EveryNoise-Watch/main/data/genre_attrs.csv"


@st.cache_data(show_spinner=False)
def load_genre_data(source: str, uploaded_bytes: Optional[bytes] = None) -> pd.DataFrame:
    """
    Loads the genre dataset.

    Supports either:
      - r/g/b columns, OR
      - a hex color column like hex_colour (e.g. "#aabbcc")
    """
    if source == "web":
        df = pd.read_csv(DATA_URL)
    else:
        if uploaded_bytes is None:
            raise ValueError("No uploaded file provided.")
        df = pd.read_csv(uploaded_bytes)

    # Must-have columns
    required_base = {"genre", "x", "y"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(f"Missing columns: {missing_base}. Found: {list(df.columns)}")

    df = df.dropna(subset=["genre", "x", "y"]).copy()
    df["genre"] = df["genre"].astype(str)

    # Ensure x/y are numeric
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).copy()

    # Color handling
    if {"r", "g", "b"}.issubset(df.columns):
        for col in ["r", "g", "b"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["r", "g", "b"]).copy()
        df["r"] = df["r"].clip(0, 255).astype(int)
        df["g"] = df["g"].clip(0, 255).astype(int)
        df["b"] = df["b"].clip(0, 255).astype(int)
    else:
        # Try common hex color column names
        hex_col = None
        for c in ["hex_colour", "hex_color", "hex", "color", "colour"]:
            if c in df.columns:
                hex_col = c
                break

        if hex_col is None:
            raise ValueError(
                f"Missing color columns. Need r/g/b or a hex column like hex_colour. Found: {list(df.columns)}"
            )

        h = df[hex_col].astype(str).str.strip().str.lstrip("#")
        # Keep only valid 6-char hex; fallback to black for invalid values
        h = h.where(h.str.len() == 6, other="000000")

        df["r"] = h.str[0:2].apply(lambda s: int(s, 16))
        df["g"] = h.str[2:4].apply(lambda s: int(s, 16))
        df["b"] = h.str[4:6].apply(lambda s: int(s, 16))

    # Keep a stable row order for indexing
    df = df.reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def build_knn_graph(coords: np.ndarray, k: int):
    """
    Builds an undirected k-nearest-neighbor graph.
    Returns:
      - adjacency list: adj[u] = [(v, distance), ...]
      - unique undirected edges: (u, v, distance) for drawing
    """
    n = coords.shape[0]
    k = max(1, min(k, n - 1))

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    edges: List[Tuple[int, int, float]] = []

    for u in range(n):
        for pos in range(1, k + 1):  # skip self at pos 0
            v = int(indices[u, pos])
            d = float(distances[u, pos])

            adj[u].append((v, d))
            adj[v].append((u, d))  # undirected

            if u < v:
                edges.append((u, v, d))

    return adj, edges


def dijkstra_path(adj: List[List[Tuple[int, float]]], start: int, goal: int) -> List[int]:
    """
    Shortest path by sum of distances. Returns node indices inclusive.
    """
    if start == goal:
        return [start]

    INF = 1e18
    dist = {start: 0.0}
    prev = {}
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


def rgb_strings(df: pd.DataFrame) -> List[str]:
    return [f"rgb({r},{g},{b})" for r, g, b in zip(df["r"], df["g"], df["b"])]


def make_figure(
    df_plot: pd.DataFrame,
    selected_idx: Optional[int],
    neighbor_idxs: Set[int],
    edges_to_draw: List[Tuple[int, int, float]],
    path: Optional[List[int]],
) -> go.Figure:
    colors = rgb_strings(df_plot)
    n = len(df_plot)

    # Default styling
    size = np.full(n, 5.0)
    opacity = np.full(n, 0.22)

    # Highlight neighbors
    for i in neighbor_idxs:
        size[i] = 9.0
        opacity[i] = 0.9

    # Highlight selected
    if selected_idx is not None:
        size[selected_idx] = 14.0
        opacity[selected_idx] = 1.0

    # Highlight path
    path_set = set(path or [])
    for i in path_set:
        size[i] = max(size[i], 11.0)
        opacity[i] = 1.0

    fig = go.Figure()

    # Lines
    if edges_to_draw:
        xs, ys = [], []
        for u, v, _d in edges_to_draw:
            xs.extend([df_plot.at[u, "x"], df_plot.at[v, "x"], None])
            ys.extend([df_plot.at[u, "y"], df_plot.at[v, "y"], None])

        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=1),
                opacity=0.30,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Points
    fig.add_trace(
        go.Scattergl(
            x=df_plot["x"],
            y=df_plot["y"],
            mode="markers",
            marker=dict(size=size, color=colors, opacity=opacity),
            text=df_plot["genre"],
            hovertemplate="%{text}<extra></extra>",
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


# ---------------- App UI ----------------
st.set_page_config(page_title="Phase 1: Genre Map Prototype", layout="wide")
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
    k = st.slider(
        "Connections per genre (nearest neighbors)",
        min_value=2,
        max_value=20,
        value=8,
        help="Higher = more connections (and more visual clutter).",
    )
    show_pathfinder = st.checkbox("Enable path finder", value=True)

    st.divider()
    st.header("View")
    view_mode = st.radio("Map shape", ["Fit to screen", "Original"], index=0)

# Use df_plot for all geometry so we can rescale without breaking indexing
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

# ---- Main layout
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Pick a genre")

    query = st.text_input("Search genres", value="")

    if query.strip():
        mask = df["genre"].str.contains(query.strip(), case=False, na=False)
        candidates = df.loc[mask, "genre"].tolist()
    else:
        candidates = df["genre"].head(200).tolist()

    if not candidates:
        st.warning("No matches. Try a different search.")
        st.stop()

    selected_genre = st.selectbox("Genre", candidates, index=0)
    selected_idx = int(df.index[df["genre"] == selected_genre][0])

    neighbor_idxs = {v for v, _d in adj[selected_idx]}
    neighbor_list = df.loc[list(neighbor_idxs), "genre"].sort_values().tolist()

    st.caption("Selected genre")
    st.markdown(f"**{selected_genre}**")

    st.caption("Closest genres")
    st.write(", ".join(neighbor_list[:25]) + (" ..." if len(neighbor_list) > 25 else ""))

    path: List[int] = []
    path_edges: List[Tuple[int, int, float]] = []

    if show_pathfinder:
        st.subheader("Path finder")

        # A filtered list helps find targets quickly
        end_query = st.text_input("Search destination genre", value="")
        if end_query.strip():
            end_mask = df["genre"].str.contains(end_query.strip(), case=False, na=False)
            end_candidates = df.loc[end_mask, "genre"].tolist()
        else:
            end_candidates = df["genre"].sample(min(250, len(df))).sort_values().tolist()

        if not end_candidates:
            st.warning("No destination matches.")
        else:
            end = st.selectbox("Find a route to:", end_candidates, index=0)

            if st.button("Find shortest path"):
                end_idx = int(df.index[df["genre"] == end][0])
                path = dijkstra_path(adj, selected_idx, end_idx)

                if path:
                    st.success(f"Found a path with {len(path)} steps.")
                    st.write([df.at[i, "genre"] for i in path[:30]] + (["…"] if len(path) > 30 else []))
                    if len(path) >= 2:
                        path_edges = [(a, b, 0.0) for a, b in zip(path[:-1], path[1:])]
                else:
                    st.error("No path found. Try increasing 'Connections per genre'.")

with right:
    st.subheader("Map")

    # Draw only a focused set of edges to keep it fast:
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
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

st.caption(
    "Data: EveryNoise-derived genre attributes (x/y + color). "
    "Prototype connects each genre to its nearest neighbors."
)
