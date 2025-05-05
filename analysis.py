from __future__ import annotations

import ast
import os
from collections import Counter
from typing import Dict, List, Set

from git import Repo
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RangeSlider
import numpy as np


REPO_FOLDER: str = "./Zeeguu-Api"
# Defines which package of the repository that will be analyzed (e.g. zeeguu/core, zeeguu/api)
PACKAGE_ROOT: str = "zeeguu/core"

if not os.path.exists(REPO_FOLDER):
    Repo.clone_from("https://github.com/zeeguu/api", REPO_FOLDER)
else:
    print(f"{REPO_FOLDER!s} already exists, skipping clone.")


ROOT_PARTS: List[str] = PACKAGE_ROOT.split("/")
PKG_ROOT_ABS: str = os.path.join(REPO_FOLDER, *ROOT_PARTS)

def get_top_level_folder(rel_path: str) -> str:
    parts = rel_path.split(os.sep)
    if parts[: len(ROOT_PARTS)] == ROOT_PARTS and len(parts) > len(ROOT_PARTS):
        return parts[len(ROOT_PARTS)]
    return rel_path

def project_top_packages() -> Set[str]:
    return {
        name
        for name in os.listdir(PKG_ROOT_ABS)
        if os.path.isdir(os.path.join(PKG_ROOT_ABS, name))
        and os.path.exists(os.path.join(PKG_ROOT_ABS, name, "__init__.py"))
    }

INTERNAL_TOP_PACKAGES: Set[str] = project_top_packages()

def find_source_files() -> List[str]:
    return [
        os.path.join(dirpath, fn)
        for dirpath, _, filenames in os.walk(PKG_ROOT_ABS)
        for fn in filenames
        if fn.endswith(".py")
    ]

def is_internal_absolute(mod_parts: List[str]) -> bool:
    return (
        len(mod_parts) >= len(ROOT_PARTS) + 1
        and mod_parts[: len(ROOT_PARTS)] == ROOT_PARTS
        and mod_parts[len(ROOT_PARTS)] in INTERNAL_TOP_PACKAGES
    )

def extract_internal_imports(file_path: str) -> List[str]:
    imports: List[str] = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read(), filename=file_path)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        # import pkg.sub … 
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_parts = alias.name.split(".")
                if is_internal_absolute(mod_parts):
                    imports.append(mod_parts[len(ROOT_PARTS)])

        # from … import … 
        elif isinstance(node, ast.ImportFrom):
            # Absolute import (level == 0)
            if node.level == 0 and node.module:
                mod_parts = node.module.split(".")
                if is_internal_absolute(mod_parts):
                    imports.append(mod_parts[len(ROOT_PARTS)])

            # Relative import
            elif node.level >= 1:
                rel_dir_parts = os.path.relpath(file_path, REPO_FOLDER).split(os.sep)[:-1]

                # Strip PACKAGE_ROOT from the front, if present
                if rel_dir_parts[: len(ROOT_PARTS)] == ROOT_PARTS:
                    rel_dir_parts = rel_dir_parts[len(ROOT_PARTS) :]

                target_parts = rel_dir_parts[: max(0, len(rel_dir_parts) - node.level)]

                if node.module:
                    target_parts += node.module.split(".")

                if target_parts:
                    dst = target_parts[0]
                    if dst in INTERNAL_TOP_PACKAGES:
                        imports.append(dst)

    return imports


print(f"Scanning {REPO_FOLDER} …")
source_files = find_source_files()
module_edges: Counter[tuple[str, str]] = Counter()
modelcount = 0
resolved_imports = total_imports = 0
for file_path in source_files:
    rel_path = os.path.relpath(file_path, REPO_FOLDER)
    src_module = get_top_level_folder(rel_path)
    dst_modules = extract_internal_imports(file_path)

    total_imports += len(dst_modules)
    for dst in dst_modules:
        if src_module != dst:
            if(src_module == "model"):
                modelcount = modelcount + 1
            resolved_imports += 1
            module_edges[(src_module, dst)] += 1

G = nx.DiGraph()
for (src, dst), w in module_edges.items():
    G.add_edge(src, dst, weight=w)


def compute_layout(graph: nx.DiGraph) -> Dict[str, np.ndarray]:
    """Return node positions with overlap avoidance & centre bias."""
    # 1. Try Graphviz (best overlap handling)
    try:
        from networkx.drawing.nx_agraph import graphviz_layout

        pos = graphviz_layout(graph, prog="neato", args="-Goverlap=false")
        # graphviz_layout returns dict[str, tuple]; convert to ndarray
        pos = {n: np.array(p, dtype=float) for n, p in pos.items()}
        # Normalise to roughly (-1,1) range
        pts = np.vstack(list(pos.values()))
        pos = {k: (v - pts.mean(0)) / pts.ptp(0).max() for k, v in pos.items()}
    except Exception:
        # Fallback: spring layout
        pos = nx.spring_layout(graph, seed=42, weight="weight", iterations=500)
        pos = {n: np.array(p) for n, p in pos.items()}

    # 2. Bias: high-degree nodes pulled toward centre
    deg = dict(graph.degree())
    if len(deg) > 1:
        min_deg, max_deg = min(deg.values()), max(deg.values())
        for n in graph:
            if max_deg != min_deg:
                # 0.6 for hubs (centre), 1.0 for leaves (outer)
                factor = 0.6 + 0.4 * (max_deg - deg[n]) / (max_deg - min_deg)
                pos[n] = pos[n] * factor

    return pos


pos = compute_layout(G)



# Derived visuals
node_sizes = [3000 + 100 * (G.out_degree(n) + G.in_degree(n)) for n in G.nodes()]
node_colors = [G.out_degree(n) for n in G.nodes()]
weights = [d["weight"] for *_, d in G.edges(data=True)]
min_w, max_w = min(weights), max(weights)

modules = sorted(G.nodes())
src_active: Dict[str, bool] = {m: True for m in modules}
dst_active: Dict[str, bool] = {m: True for m in modules}

# ─────────────────────────────────────────────────────────────────────────────
# 6 .  Figure & widgets
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
ax_graph = fig.add_axes([0.05, 0.05, 0.58, 0.9])

ax_src = fig.add_axes([0.68, 0.55, 0.28, 0.4])
ax_src.set_title("Show edges FROM (source)", fontsize=10)
check_src = CheckButtons(ax_src, modules, [True] * len(modules))

ax_dst = fig.add_axes([0.68, 0.10, 0.28, 0.4])
ax_dst.set_title("Show edges TO (destination)", fontsize=10)
check_dst = CheckButtons(ax_dst, modules, [True] * len(modules))

ax_slider = fig.add_axes([0.05, 0.95, 0.58, 0.03])
slider = RangeSlider(ax_slider, "Weight range", min_w, max_w, valinit=(min_w, max_w), valstep=1)

ax_btn_src_show = fig.add_axes([0.68, 0.51, 0.13, 0.03])
ax_btn_src_hide = fig.add_axes([0.83, 0.51, 0.13, 0.03])
ax_btn_dst_show = fig.add_axes([0.68, 0.06, 0.13, 0.03])
ax_btn_dst_hide = fig.add_axes([0.83, 0.06, 0.13, 0.03])

btn_src_show = Button(ax_btn_src_show, "Show ALL")
btn_src_hide = Button(ax_btn_src_hide, "Hide ALL")
btn_dst_show = Button(ax_btn_dst_show, "Show ALL")
btn_dst_hide = Button(ax_btn_dst_hide, "Hide ALL")


def _draw_graph() -> None:
    ax_graph.clear()
    w_min, w_max = slider.val

    visible = [
        (u, v, d["weight"])
        for u, v, d in G.edges(data=True)
        if src_active[u] and dst_active[v] and w_min <= d["weight"] <= w_max
    ]

    if not visible:
        ax_graph.text(
            0.5,
            0.5,
            "No edges to display",
            ha="center",
            va="center",
            fontsize=14,
            color="crimson",
        )
        ax_graph.axis("off")
        fig.canvas.draw_idle()
        return

    H = nx.DiGraph()
    for u, v, w in visible:
        H.add_edge(u, v, weight=w)

    sub_nodes = list(H.nodes())
    idx = {n: list(G.nodes()).index(n) for n in sub_nodes}
    sub_node_sizes = [node_sizes[idx[n]] for n in sub_nodes]
    sub_node_colors = [node_colors[idx[n]] for n in sub_nodes]
    sub_edge_w = [1 + 3 * w / max(w for *_, w in visible) for *_, w in visible]

    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=sub_nodes,
        node_size=sub_node_sizes,
        node_color=sub_node_colors,
        cmap=plt.cm.viridis,
        ax=ax_graph,
    )
    nx.draw_networkx_edges(
        H,
        pos,
        width=sub_edge_w,
        edge_color="gray",
        arrows=True,
        arrowsize=15,
        ax=ax_graph,
        node_size=sub_node_sizes,
    )
    nx.draw_networkx_labels(H, pos, font_size=12, font_weight="bold", ax=ax_graph)

    # ─ Edge-label placement ───────────────────────────────────────────────
    vis_set = {(u, v) for u, v, _ in visible}
    lbl_fwd, lbl_bwd, lbl_single = {}, {}, {}
    for u, v, w in visible:
        if (v, u) in vis_set:
            (lbl_fwd if u < v else lbl_bwd)[(u, v)] = str(w)
        else:
            lbl_single[(u, v)] = str(w)

    nx.draw_networkx_edge_labels(H, pos, edge_labels=lbl_fwd, font_size=8, label_pos=0.3, ax=ax_graph)
    nx.draw_networkx_edge_labels(H, pos, edge_labels=lbl_bwd, font_size=8, label_pos=0.7, ax=ax_graph)
    nx.draw_networkx_edge_labels(H, pos, edge_labels=lbl_single, font_size=8, label_pos=0.5, ax=ax_graph)

    ax_graph.set_title(
        f"{PACKAGE_ROOT} – Package Dependency Graph", fontsize=16
    )
    ax_graph.axis("off")
    fig.canvas.draw_idle()


def _toggle_src(label: str):
    src_active[label] = not src_active[label]
    _draw_graph()


def _toggle_dst(label: str):
    dst_active[label] = not dst_active[label]
    _draw_graph()


def _update_slider(_):
    _draw_graph()


def _set_all_src(state: bool, *_):
    for i, mod in enumerate(modules):
        if src_active[mod] != state:
            check_src.set_active(i)  # triggers redraw via callback


def _set_all_dst(state: bool, *_):
    for i, mod in enumerate(modules):
        if dst_active[mod] != state:
            check_dst.set_active(i)


check_src.on_clicked(_toggle_src)
check_dst.on_clicked(_toggle_dst)
slider.on_changed(_update_slider)

btn_src_show.on_clicked(lambda event: _set_all_src(True))
btn_src_hide.on_clicked(lambda event: _set_all_src(False))
btn_dst_show.on_clicked(lambda event: _set_all_dst(True))
btn_dst_hide.on_clicked(lambda event: _set_all_dst(False))

_draw_graph()
plt.show()