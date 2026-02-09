import numpy as np
import networkx as nx
import plotly.graph_objects as go

def plot_issue_network(tfidf_results: dict, issue: str, top_k: int = 30, weight_col: str = "diff", seed: int = 123):
    top_pos = tfidf_results[issue]["top_pos"].copy().head(top_k)
    top_pos = top_pos[top_pos[weight_col] > 0].copy()
    if top_pos.empty:
        raise ValueError(f"No positive weights to plot for issue={issue}")

    G = nx.Graph()
    center = issue
    G.add_node(center, kind="issue", weight=float(top_pos[weight_col].max()))

    for _, row in top_pos.iterrows():
        term = row["term"]
        w = float(row[weight_col])
        G.add_node(term, kind="term", weight=w)
        G.add_edge(center, term, weight=w)

    pos = nx.spring_layout(G, seed=seed, k=0.7)

    edge_traces = []
    edge_w = [data["weight"] for _, _, data in G.edges(data=True)]
    ew = np.array(edge_w)
    ew_norm = 1 + 8 * (ew - ew.min()) / (ew.max() - ew.min() + 1e-9)

    for (u, v, data), width in zip(G.edges(data=True), ew_norm):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=float(width)),
                hoverinfo="text",
                text=f"{u} ↔ {v}<br>{weight_col}={data['weight']:.4f}",
                showlegend=False,
            )
        )

    node_x, node_y, node_text, hover_text, node_size, node_is_center = [], [], [], [], [], []
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        w = float(data.get("weight", 0.0))
        node_size.append(w)
        node_is_center.append(n == center)
        node_text.append(n.replace("_", " ") if n == center else n)
        hover_text.append(f"{n}<br>{weight_col}={w:.4f}")

    ns = np.array(node_size)
    ns_norm = 18 + 45 * (ns - ns.min()) / (ns.max() - ns.min() + 1e-9)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=12),
        hoverinfo="text",
        hovertext=hover_text,
        marker=dict(
            size=ns_norm,
            symbol=["diamond" if c else "circle" for c in node_is_center],
        ),
        showlegend=False,
    )

    fig = go.Figure(edge_traces + [node_trace])
    fig.update_layout(
        title=f"TF-IDF Contrast Network: {issue} (top {len(top_pos)} terms)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.show()
