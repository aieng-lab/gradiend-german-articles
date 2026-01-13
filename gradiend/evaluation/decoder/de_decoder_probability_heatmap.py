# -*- coding: utf-8 -*-

"""
Heatmap (ds x article) of mean deltas with 3 significance hatch levels
and optional "expected cells" highlighting derived from GradiendGenderCaseConfiguration.

Assumes you already exported a long-form CSV like:
  columns: ds, article, mean_delta, p
optional: n, config_id, source_article, ...

Example:
  python plot_article_heatmap.py \
    --cells_csv results/decoder/article_x_ds_cells_long.csv \
    --out_png  results/decoder/article_x_ds_heatmap.png \
    --signif_mode fdr \
    --alpha 0.05 \
    --articles der die das den dem des \
    --expected_from_config

If you use --expected_from_config, this script imports your GrADIEND enums/mappings.
"""

from __future__ import annotations
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import os
import pandas as pd

from gradiend.evaluation.decoder.de_decoder_probability_analysis import _get_token_col, \
    paired_effect, determine_best_gf_alpha
from gradiend.evaluation.xai.io import all_interesting_classes, article_mapping, GradiendGenderCaseConfiguration, \
    pretty_dataset_mapping, statistical_analysis_config_classes, pretty_model_mapping
from gradiend.util import init_matplotlib, IMAGE_DIR

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

EXPECTED_EDGE_COLORS = {
    "decrease": "#2166ac",  # dark blue
    "increase": "#b2182b",  # dark red
}



# ----------------------------
# Stats helpers
# ----------------------------
def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg q-values for an array of p-values (NaNs allowed).
    Returns array of same shape with q-values (NaN where p is NaN).
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    mask = ~np.isnan(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return q

    order = np.argsort(pv)
    pv_sorted = pv[order]
    ranks = np.arange(1, m + 1, dtype=float)

    q_sorted = pv_sorted * m / ranks
    # enforce monotonicity from end
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    qv = q_sorted[inv]

    q[mask] = qv
    return q


def value_to_sig_level(x: float, thresholds=(0.05, 0.01, 0.001)) -> int:
    """
    Map p or q value -> significance level:
      3 if x < 0.001
      2 if x < 0.01
      1 if x < 0.05
      0 otherwise
    """
    if np.isnan(x):
        return 0
    a05, a01, a001 = thresholds
    if x < a001:
        return 3
    if x < a01:
        return 2
    if x < a05:
        return 1
    return 0


# ----------------------------
# Plotting
# ----------------------------

@dataclass
class HeatmapStyle:
    cmap: str = "coolwarm"
    hatch_map: Dict[int, str] = None
    hatch_edgecolor: str = "black"
    expected_edgecolor: str = "black"
    expected_linewidth: float = 2.0
    grid_linewidth: float = 0.5
    colorbar_label: str = "$\Delta \mathbb{P}(article)$"
def _add_expected_direction_boxes(
    ax,
    expected_cells,          # list of (ds, article, direction)
    ds_labels,
    art_labels,
    train_datasets=None,
):
    """
    Draw expected-direction boxes.
    Training datasets are highlighted with thicker borders.
    """

    if train_datasets is None:
        train_datasets = set()

    cell_index = {
        (ds, art): (i, j)
        for i, ds in enumerate(ds_labels)
        for j, art in enumerate(art_labels)
    }

    for ds, art, direction in expected_cells:
        if (ds, art) not in cell_index:
            continue

        i, j = cell_index[(ds, art)]

        # thicker border for training datasets
        lw = 3.2 if ds in train_datasets else 2.0
        lw = 4.0
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor=EXPECTED_EDGE_COLORS[direction],
                linewidth=lw,
            )
        )


def _add_hatching_cells(ax, hatch_cells, ds_labels, art_labels, *, hatch="///", colors=None):
    colors = colors or {}
    cell_index = {(ds, art): (i, j)
                  for i, ds in enumerate(ds_labels)
                  for j, art in enumerate(art_labels)}

    for ds, art in hatch_cells:
        if (ds, art) not in cell_index:
            continue
        i, j = cell_index[(ds, art)]
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1, 1,
                fill=False,
                hatch=hatch,
                edgecolor=colors.get((ds, art), "black"),
                linewidth=0.0,
            )
        )


def _add_sig_hatches(ax, level_mat: np.ndarray, style: HeatmapStyle):
    n_rows, n_cols = level_mat.shape
    # draw strongest first
    for i in range(n_rows):
        for j in range(n_cols):
            lvl = level_mat[i, j]
            if lvl in style.hatch_map:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch=style.hatch_map[lvl],
                        edgecolor="black",
                        linewidth=0.0,
                    )
                )

def _add_sig_stars(ax, level_mat: np.ndarray, *, fontsize: float = 7.0):
    n_rows, n_cols = level_mat.shape
    for i in range(n_rows):
        for j in range(n_cols):
            lvl = int(level_mat[i, j])
            if lvl <= 0:
                continue
            stars = "*" * lvl  # 1..3
            # top-right corner (inside cell)
            ax.text(
                j + 0.48, i - 0.48,
                stars,
                ha="right", va="top",
                fontsize=fontsize,
                color="black",
            )



def _filter_on_alpha_gf(df: pd.DataFrame, *, alpha=None, gf=None) -> pd.DataFrame:
    """Filter df if alpha/gf exist; otherwise return df unchanged."""
    out = df
    if alpha is not None and "alpha" in out.columns:
        out = out[out["alpha"] == alpha]
    if gf is not None and "gf" in out.columns:
        out = out[out["gf"] == gf]
    return out.reset_index(drop=True)

def _add_sig_legend(
    ax,
    style: HeatmapStyle,
    signif_mode: str,                 # "p" or "fdr"
    thresholds=(0.05, 0.01, 0.001),
    loc="upper left",
):
    """
    Add a compact significance legend.
    Labels adapt automatically to signif_mode ("p" or "fdr").
    """

    if signif_mode == "fdr":
        prefix = "q"
    elif signif_mode == "p":
        prefix = "p"
    else:
        raise ValueError("signif_mode must be 'p' or 'fdr'")

    legend_hatch_map = style.hatch_map
    handles, labels = [], []
    for lvl, thr in [(3, thresholds[2]), (2, thresholds[1]), (1, thresholds[0])]:
        handles.append(
            Rectangle(
                (0, 0), 1, 1,
                fill=False,
                hatch=legend_hatch_map[lvl],
                edgecolor="black",
                linewidth=0.8,
            )
        )
        labels.append(f"{prefix} $< {thr:g}$")

    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        handleheight=1.2,
        columnspacing=1.0,
        fontsize=8,
    )



class HandlerSquarePatch(HandlerBase):
    """Draw the given Rectangle proxy as a centered SQUARE inside the legend handle box."""
    def create_artists(self, legend, orig_handle, x0, y0, w, h, fontsize, trans):
        s = min(w, h)
        xs = x0 + (w - s) / 2.0
        y_shift = -0.19 * h
        ys = y0 + (h - s) / 2.0 + y_shift

        r = Rectangle(
            (xs, ys), s, s,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            linewidth=orig_handle.get_linewidth(),
            hatch=orig_handle.get_hatch(),
            fill=orig_handle.get_fill(),
            transform=trans,
        )
        return [r]

class HandlerSquareWithLowerLeftTriangle(HandlerBase):
    """Draw a square cell context + lower-left filled triangle inside it."""
    def __init__(self, *, tri_color="green", tri_alpha=0.25,
                 cell_edge="black", cell_lw=0.8):
        super().__init__()
        self.tri_color = tri_color
        self.tri_alpha = tri_alpha
        self.cell_edge = cell_edge
        self.cell_lw = cell_lw

    def create_artists(self, legend, orig_handle, x0, y0, w, h, fontsize, trans):
        s = min(w, h)
        xs = x0 + (w - s) / 2.0
        y_shift = -0.15 * h
        ys = y0 + (h - s) / 2.0 + y_shift

        # square "cell" context
        cell = Rectangle(
            (xs, ys), s, s,
            facecolor="none",
            edgecolor=self.cell_edge,
            linewidth=self.cell_lw,
            transform=trans,
        )

        # lower-left triangle (same geometry as in your heatmap)
        tri = Polygon(
            [(xs, ys + s), (xs, ys), (xs + s, ys)],
            closed=True,
            facecolor=self.tri_color,
            edgecolor="none",
            alpha=self.tri_alpha,
            transform=trans,
        )
        return [cell, tri]

def add_highlight_legend(ax, *,
                         hatch="////",
                         local_color="black",
                         local_alpha=0.4,
                         outline_color="#2166ac",
                         outline_lw=2.0,
                         fontsize=12):
    # proxies
    local_proxy = Rectangle((0, 0), 1, 1, fill=False, edgecolor="none")  # drawn by handler
    mem_proxy   = Rectangle((0, 0), 1, 1, fill=False, hatch=hatch, edgecolor="black", linewidth=0.8)
    abs_proxy   = Rectangle((0, 0), 1, 1, fill=False, edgecolor=outline_color, linewidth=outline_lw)

    ax.legend(
        handles=[local_proxy, abs_proxy, mem_proxy],
        labels=["LR", "GR", "SO"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.2,   # keep compact
        handleheight=1.2,   # make handles square-ish
        columnspacing=1.2,
        handler_map={
            local_proxy: HandlerSquareWithLowerLeftTriangle(
                tri_color=local_color, tri_alpha=local_alpha,
                cell_edge="black", cell_lw=0.8
            ),
            mem_proxy: HandlerSquarePatch(),
            abs_proxy: HandlerSquarePatch(),
        }
    )



from matplotlib.patches import Rectangle
import numpy as np

def _add_corner_markers(ax, cells, ds_labels, art_labels, *, corner="br", size=18, color="black"):
    # Map (row_label, col_label) -> (i, j)
    idx = {(ds, art): (i, j)
           for i, ds in enumerate(ds_labels)
           for j, art in enumerate(art_labels)}

    # offsets inside a cell (corners)
    corner_xy = {
        "tr": (0.46, -0.46),
        "tl": (-0.46, -0.46),
        "br": (0.46, 0.46),
        "bl": (-0.46, 0.46),
    }
    dx, dy = corner_xy[corner]

    xs, ys = [], []
    for key in cells:
        if key not in idx:
            continue
        i, j = idx[key]
        xs.append(j + dx)
        ys.append(i + dy)

    if xs:
        ax.scatter(xs, ys, s=size, marker="o", linewidths=0, c=color)  # default black

from matplotlib.patches import Polygon, Patch

def _add_local_triangles(ax, local_cells, ds_labels, art_labels, *, colors=None, alpha=0.4):
    colors = colors or {}
    idx = {(ds, art): (i, j)
           for i, ds in enumerate(ds_labels)
           for j, art in enumerate(art_labels)}

    for key in local_cells:
        if key not in idx:
            continue
        i, j = idx[key]

        # cell corners (imshow-style): center=(j,i), extents +/- 0.5
        xL, xR = j - 0.5, j + 0.5
        yT, yB = i - 0.5, i + 0.5

        # lower-left triangle (below diagonal top-left -> bottom-right)
        tri = Polygon([(xL, yT), (xL, yB), (xR, yB)],
                      closed=True, facecolor=colors.get(key, "black"), edgecolor="none", alpha=alpha, zorder=3)
        ax.add_patch(tri)

def _add_local_rectangles(ax, local_cells, ds_labels, art_labels, *, color="black", alpha=0.4):
    idx = {(ds, art): (i, j)
           for i, ds in enumerate(ds_labels)
           for j, art in enumerate(art_labels)}

    for key in local_cells:
        if key not in idx:
            continue
        i, j = idx[key]

        # cell corners (imshow-style): center=(j,i), extents +/- 0.5
        xL, xR = j - 0.5, j + 0.5
        yT, yB = i - 0.5, i + 0.5

        # full rectangle
        rect = Rectangle((xL, yT), width=1.0, height=1.0,
                         facecolor=color, edgecolor="none", alpha=alpha, zorder=3)
        ax.add_patch(rect)



def plot_article_delta_heatmap(
    cells_df: pd.DataFrame,
    *,
    articles_order: Optional[Sequence[str]] = None,
    ds_order: Optional[Sequence[str]] = None,
    agg: str = "mean",                    # mean or median
    signif_mode: str = "fdr",               # "p" or "fdr"
    thresholds: Tuple[float, float, float] = (0.05, 0.01, 0.001),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    expected_cells: Optional[List[Tuple[str, str]]] = None,  # list of (ds, article)
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (11, 7),
    style: Optional[HeatmapStyle] = None,
    train_datasets: Optional[Iterable[str]] = None,
    hatch_cells: Optional[List[Tuple[str, str]]] = None,
    other_hatch_cells: Optional[List[Tuple[str, str]]] = None,
    hatch_pattern: str = "///",
    other_hatch_pattern: str = "\\\\\\",
    corner_cells: Optional[List[Tuple[str, str]]] = None,
):
    """
    cells_df must contain columns: ds, article, mean_delta, p
    Optional: multiple rows per (ds, article) aggregated.
    """
    style = style or HeatmapStyle(
        hatch_map={
            #1: ".",     # p/q < 0.05
            2: "//",    # p/q < 0.01
            3: "//",    # p/q < 0.001
        }
    )

    expected_cells = expected_cells or []
    required = {"ds", "article", "mean_delta", "p_perm"}
    missing = required - set(cells_df.columns)
    if missing:
        raise ValueError(f"cells_df missing columns: {missing}")

    # Aggregate if duplicates per (ds, article) exist
    if agg == "mean":
        grouped = cells_df.groupby(["ds", "article"], as_index=False).agg(
            mean_delta=("mean_delta", "mean"),
            p=("p_perm", "mean"),
            n=("n", "sum") if "n" in cells_df.columns else ("mean_delta", "size"),
        )
    elif agg == "median":
        grouped = cells_df.groupby(["ds", "article"], as_index=False).agg(
            mean_delta=("mean_delta", "median"),
            p=("p_perm", "median"),
            n=("n", "sum") if "n" in cells_df.columns else ("mean_delta", "size"),
        )
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    M = grouped.pivot(index="ds", columns="article", values="mean_delta")
    P = grouped.pivot(index="ds", columns="article", values="p")

    if ds_order is not None:
        M = M.reindex(ds_order)
        P = P.reindex(ds_order)
    if articles_order is not None:
        M = M.reindex(columns=list(articles_order))
        P = P.reindex(columns=list(articles_order))

    ds_labels = list(M.index)
    art_labels = list(M.columns)
    hatch_cells = hatch_cells or []
    other_hatch_cells = other_hatch_cells or []


    data = M.to_numpy(dtype=float)
    pmat = P.to_numpy(dtype=float)

    data = data * 100.0  # convert to percentage points

    # choose scale
    if vmin is None or vmax is None:
        absmax = np.nanmax(np.abs(data)) if np.isfinite(np.nanmax(np.abs(data))) else 1.0
        if vmin is None:
            vmin = -absmax
        if vmax is None:
            vmax = absmax

    # compute significance values (p or q)
    if signif_mode == "p":
        sig_values = pmat
    elif signif_mode == "fdr":
        sig_values = bh_qvalues(pmat)
    else:
        raise ValueError("signif_mode must be 'p' or 'fdr'")

    level_mat = np.vectorize(lambda x: value_to_sig_level(x, thresholds=thresholds))(sig_values).astype(int)

    fig, ax = plt.subplots(figsize=figsize)
    n_rows, n_cols = data.shape
    cell_size = 0.47
    fig.set_size_inches(n_cols * cell_size, n_rows * cell_size)
    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap=style.cmap)
    ax.set_aspect("equal", adjustable="box")

    # ticks / labels
    ax.set_xticks(np.arange(len(art_labels)))
    ax.set_yticks(np.arange(len(ds_labels)))
    ax.set_xticklabels(art_labels, rotation=45, ha="right")
    ax.set_yticklabels(ds_labels)

    # grid
    ax.set_xticks(np.arange(-0.5, len(art_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ds_labels), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=style.grid_linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Article", fontsize=9)
    ax.set_ylabel("Dataset", fontsize=9)

    ax.set_yticklabels([
        f"{pretty_dataset_mapping.get(ds, ds)}$({article_mapping.get(ds, ds)})$"
        for ds in ds_labels
    ], fontsize=12)

    # hatches for significance (3 types)
    #_add_sig_hatches(ax, level_mat, style)
    _add_sig_stars(ax, level_mat, fontsize=5.0)

    # highlight expected cells (thick outline)
    _add_expected_direction_boxes(
        ax,
        expected_cells=expected_cells,
        ds_labels=ds_labels,
        art_labels=art_labels,
        train_datasets=train_datasets,
    )
    _add_hatching_cells(ax, hatch_cells, ds_labels, art_labels, hatch=hatch_pattern)
    _add_hatching_cells(ax, other_hatch_cells, ds_labels, art_labels, hatch=other_hatch_pattern)

    #_add_corner_markers(ax, corner_cells, ds_labels, art_labels, corner="br", size=16)
    _add_local_triangles(ax, corner_cells or [], ds_labels, art_labels, alpha=0.25)
    #_add_local_rectangles(ax, corner_cells or [], ds_labels, art_labels, color="black", alpha=0.25)

    from matplotlib.ticker import FuncFormatter

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.06,
        pad=0.02,
        aspect=30,
    )

    cbar.set_label(r"$\Delta \mathbb{P}(article)$", fontsize=8)

    # make ticks not stick out beyond the bar
    cbar.ax.tick_params(
        labelsize=7,
        length=2.0,  # shorter ticks
        width=0.8,
        direction="in",  # draw ticks into the colorbar
        pad=1.0
    )

    # percent tick labels (no manual set_ticklabels)
    cbar.formatter = FuncFormatter(lambda x, pos: f"{x:.2f}\%")
    cbar.update_ticks()

    add_highlight_legend(ax, outline_color=EXPECTED_EDGE_COLORS["increase"])

    if title:
        ax.set_title(title)

    #fig.tight_layout()
    return fig, ax


# ----------------------------
# Expected datasets / cells from config
# ----------------------------

def invert_enum2id(enum2id_map, all_cases, all_genders):
    """
    Build inverse lookup from your enum2id mapping.
    Robust to multi-char ids via length-sorted matching.
    """
    id2case = {enum2id_map[c]: c for c in all_cases}
    id2gender = {enum2id_map[g]: g for g in all_genders}
    case_codes = sorted(id2case.keys(), key=len, reverse=True)
    gender_codes = sorted(id2gender.keys(), key=len, reverse=True)
    return id2case, id2gender, case_codes, gender_codes


def parse_ds(ds: str, id2case, id2gender, case_codes, gender_codes):
    case_code = None
    for cc in case_codes:
        if ds.startswith(cc):
            case_code = cc
            break
    if case_code is None:
        raise ValueError(f"Could not parse case from ds='{ds}'")
    rest = ds[len(case_code):]
    gender_code = None
    for gc in gender_codes:
        if rest == gc:
            gender_code = gc
            break
    if gender_code is None:
        raise ValueError(f"Could not parse gender from ds='{ds}' (rest='{rest}')")
    return id2case[case_code], id2gender[gender_code], case_code, gender_code



def analyze_steering_all_articles_from_files(
    *,
    base_eval_csv: str,
    mod_eval_csv: str,
    base_neutral_csv: str,
    mod_neutral_csv: str,
    select_token: str,                 # token used ONLY to select gf/alpha (e.g. other_article)
    TG,
    articles_to_score,                 # list like ["der","die","das","den","dem","des"]
    accuracy_threshold: float = 0.99,
    base_alpha: int = 0,               # baseline alpha/gf if present
    base_gf: int = 0,
):
    """
    Compute ds x article stats (mean_delta, p, etc.) for ALL requested articles,
    using gf/alpha selected based on `select_token`.

    Returns:
      cells_df: long-form with columns [ds, article, mean_delta, p, n, t, cohen_d]
      gf, alpha: selected steering params
    """
    base_eval = pd.read_csv(base_eval_csv)
    mod_eval = pd.read_csv(mod_eval_csv)
    base_neutral = pd.read_csv(base_neutral_csv)
    mod_neutral = pd.read_csv(mod_neutral_csv)

    # pick gf/alpha using your existing selection logic (based on select_token)
    gf, alpha = determine_best_gf_alpha(
        mod_eval=mod_eval,
        mod_neutral=mod_neutral,
        TG=list(TG),
        target_token=select_token,
        accuracy_threshold=accuracy_threshold,
    )

    # filter mod to selected steering
    mod_eval = _filter_on_alpha_gf(mod_eval, alpha=alpha, gf=gf)
    mod_neutral = _filter_on_alpha_gf(mod_neutral, alpha=alpha, gf=gf)

    if mod_eval.empty:
        raise ValueError(f"mod_eval empty after filtering to alpha={alpha}, gf={gf}")

    # filter baseline if it contains alpha/gf
    base_eval = _filter_on_alpha_gf(base_eval, alpha=base_alpha, gf=base_gf)
    base_neutral = _filter_on_alpha_gf(base_neutral, alpha=base_alpha, gf=base_gf)

    if "ds" not in base_eval.columns or "ds" not in mod_eval.columns:
        raise ValueError("base_eval and mod_eval must contain 'ds' column")

    # IMPORTANT: ensure paired alignment per ds
    # If your evaluation guaranteed same row order, this is enough.
    # Otherwise, you must merge on a stable key (e.g. sentence_id).
    # Here we follow your existing assumption (order-aligned).
    cells_rows = []
    for ds in sorted(base_eval["ds"].unique()):
        base_ds = base_eval[base_eval["ds"] == ds].reset_index(drop=True)
        mod_ds = mod_eval[mod_eval["ds"] == ds].reset_index(drop=True)

        # avoid accidental mismatch
        n_ds = min(len(base_ds), len(mod_ds))
        if n_ds < 2:
            continue
        base_ds = base_ds.iloc[:n_ds]
        mod_ds = mod_ds.iloc[:n_ds]

        for tok in articles_to_score:
            try:
                col_b = _get_token_col(base_ds, tok)
                col_m = _get_token_col(mod_ds, tok)
            except ValueError:
                # token column not present in your CSV -> skip
                continue

            stats = paired_effect(base_ds[col_b].to_numpy(), mod_ds[col_m].to_numpy())
            stats.update({"ds": ds, "article": tok})
            cells_rows.append(stats)

    cells_df = pd.DataFrame(cells_rows)
    return cells_df, gf, alpha



def run_heatmap_like_your_main(
    *,
    base_model_id: str,
    all_interesting_classes: dict,
    articles_order=("der", "die", "das", "den", "dem", "des"),
    results_decoder_dir: str = "results/decoder",
    out_dir: str = None,
    signif_mode: str = "fdr",
    thresholds=(0.05, 0.01, 0.001),
    rule_axis: str = "gender",
    highlight_expected: bool = True,
    restrict_expected_to_config: bool = False,
    accuracy_threshold: float = 0.99,
    verbose: bool = True,
):
    """
    Mirrors your loop but produces a FULL ds x article heatmap per (config, article->other_article),
    i.e. mean_delta computed for each article in `articles_order`.
    """

    from gradiend.evaluation.xai.io import ALL_CASES, ALL_GENDERS, enum2id, case_gender_mapping
    init_matplotlib(use_tex=True)

    if out_dir is None:
        out_dir = f"{IMAGE_DIR}/decoder_heatmaps/{base_model_id}"
    os.makedirs(out_dir, exist_ok=True)

    for articles, config_keys in statistical_analysis_config_classes.items():
        configs = [GradiendGenderCaseConfiguration(*config_key, model_id=base_model_id) for config_key in config_keys]
        datasets_union = list(set.union(*[set(cfg.datasets) for cfg in configs]))

        for config in configs:
            try:
                model_path = config.gradiend_dir
            except Exception:
                print(f"[ERROR] Could not get model path for config {config.id}")
                continue
            config_datasets = config.datasets

            for article in config.articles:
                other_article = [a for a in config.articles if a != article][0]

                this_dataset = [ds for ds in config_datasets if article_mapping[ds] == article]
                other_configs_datasets_with_same_article = [
                    ds for ds in datasets_union if article_mapping[ds] == article and ds not in this_dataset
                ]
                other_article_datasets = [
                    ds for ds, art in article_mapping.items() if art not in config.articles
                ]

                try:
                    # compute ds x ALL-articles stats (this is the key fix)
                    cells_df, gf, alpha = analyze_steering_all_articles_from_files(
                        base_eval_csv=f"{results_decoder_dir}/{base_model_id}_base_eval.csv",
                        mod_eval_csv=f"{model_path}_decoder_sig_analysis_mod_eval.csv",
                        base_neutral_csv=f"{results_decoder_dir}/{base_model_id}_base_neutral.csv",
                        mod_neutral_csv=f"{model_path}_decoder_sig_analysis_mod_neutral.csv",
                        select_token=other_article,              # selection only
                        TG=this_dataset,
                        articles_to_score=list(articles_order),  # compute deltas for each
                        accuracy_threshold=accuracy_threshold,
                    )

                    if cells_df.empty:
                        raise ValueError("cells_df empty (no ds/article stats computed)")

                    # expected cells highlight (for the steered-to article)
                    abstract_rule_based_cells = config.abstract_rule_based_gender_case_article_pairs
                    hatch_cells = config.memorization_gender_case_article_pairs[article]
                    corner_cells = config.local_rule_based_gender_case_article_pairs

                    title = rf"{base_model_id} | {config.id} | $\alpha$={alpha}, gf={gf} | steer {article}->{other_article}"
                    print(title)
                    fig, ax = plot_article_delta_heatmap(
                        cells_df,                              # long-form: ds, article, mean_delta, p
                        articles_order=list(articles_order),
                        ds_order=list(article_mapping.keys()),
                        agg="mean",
                        signif_mode=signif_mode,
                        thresholds=thresholds,
                        expected_cells=abstract_rule_based_cells,  # outlines (abstract)
                        hatch_cells=hatch_cells,  # hatching (memorization)
                        #other_hatch_cells=other_hatch_cells,
                        figsize=(7, 6),
                        train_datasets=config_datasets,
                        corner_cells=corner_cells,
                    )
                    fig.subplots_adjust(bottom=0.22)
                    out_png = os.path.join(out_dir, f"{config.id}__{article}_to_{other_article}__full.pdf")

                    fig.savefig(out_png, dpi=300) #, bbox_inches="tight")

                    plt.show()
                    plt.close(fig)

                    if verbose:
                        print(f"Saved: {out_png}")

                except NotImplementedError as e:
                    print(f"[ERROR] {base_model_id} | {config.id} | {article}->{other_article}: {e}")



def _prepare_heatmap_mats(
    cells_df: pd.DataFrame,
    *,
    articles_order: Optional[Sequence[str]],
    ds_order: Optional[Sequence[str]],
    agg: str,
    signif_mode: str,
    thresholds: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Returns: data(%), level_mat, ds_labels, art_labels."""
    required = {"ds", "article", "mean_delta", "p_perm"}
    missing = required - set(cells_df.columns)
    if missing:
        raise ValueError(f"cells_df missing columns: {missing}")

    if agg == "mean":
        grouped = cells_df.groupby(["ds", "article"], as_index=False).agg(
            mean_delta=("mean_delta", "mean"),
            p=("p_perm", "mean"),
        )
    elif agg == "median":
        grouped = cells_df.groupby(["ds", "article"], as_index=False).agg(
            mean_delta=("mean_delta", "median"),
            p=("p_perm", "median"),
        )
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    M = grouped.pivot(index="ds", columns="article", values="mean_delta")
    P = grouped.pivot(index="ds", columns="article", values="p")

    if ds_order is not None:
        M = M.reindex(ds_order)
        P = P.reindex(ds_order)
    if articles_order is not None:
        M = M.reindex(columns=list(articles_order))
        P = P.reindex(columns=list(articles_order))

    ds_labels = list(M.index)
    art_labels = list(M.columns)

    data = M.to_numpy(dtype=float) * 100.0  # percentage points
    pmat = P.to_numpy(dtype=float)

    if signif_mode == "p":
        sig_values = pmat
    elif signif_mode == "fdr":
        sig_values = bh_qvalues(pmat)
    else:
        raise ValueError("signif_mode must be 'p' or 'fdr'")

    level_mat = np.vectorize(lambda x: value_to_sig_level(x, thresholds=thresholds))(sig_values).astype(int)
    return data, level_mat, ds_labels, art_labels


def add_highlight_legend_fig(fig, *,
                             hatch="////",
                             outline_color="black",
                             outline_lw=2.0,
                             fontsize=12,
                             ):
    # proxies
    local_proxy = Rectangle((0, 0), 1, 1, fill=False, hatch="\\\\\\", edgecolor="black", linewidth=0.8)
    mem_proxy   = Rectangle((0, 0), 1, 1, fill=False, hatch=hatch, edgecolor="black", linewidth=0.8)
    abs_proxy   = Rectangle((0, 0), 1, 1, fill=False, edgecolor=outline_color, linewidth=outline_lw)

    fig.legend(
        handles=[local_proxy, abs_proxy, mem_proxy],
        labels=["LR", "GR", "SO"],
        #loc="upper center",
        loc="lower left",
        #bbox_to_anchor=(0.5, y),
        bbox_to_anchor=(0.04, 0.03),
        ncol=1,
        frameon=True,
        fontsize=fontsize,
        handlelength=1.2,
        handleheight=1.2,
        columnspacing=1.2,
        handler_map={
            #local_proxy: HandlerSquareWithLowerLeftTriangle(
            #    tri_color=local_color, tri_alpha=local_alpha,
            #    cell_edge="black", cell_lw=0.8
            #),
            local_proxy: HandlerSquarePatch(),
            mem_proxy: HandlerSquarePatch(),
            abs_proxy: HandlerSquarePatch(),
        }
    )


def plot_article_delta_heatmap_multi_models(
    model_to_cells_df: Dict[str, pd.DataFrame],
    *,
    articles_order: Optional[Sequence[str]] = None,
    ds_order: Optional[Sequence[str]] = None,
    agg: str = "mean",
    signif_mode: str = "fdr",
    thresholds: Tuple[float, float, float] = (0.05, 0.01, 0.001),
    style: Optional[HeatmapStyle] = None,
    # highlights can be shared or per-model
    model_to_expected_cells: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
    model_to_hatch_cells: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    model_to_other_hatch_cells: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    model_to_local_cells: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    model_to_train_datasets: Optional[Dict[str, Iterable[str]]] = None,
    # layout
    model_order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    cell_size: float = 0.42,    # tweak for compactness
    cbar_mode: str = "inside",  # "inside" | "right" | "below"
    cbar_frac_right: float = 0.06,
    cbar_pad_right: float = 0.02,
    cbar_aspect_right: float = 30.0,
    cbar_height_below: float = 0.02,  # fraction of axes height when below
    cbar_pad_below: float = 0.23,  # pad between heatmap and below-cbar (in axes fraction)
    show_legend: bool = True,
    target_article=None,
):
    """
    Single figure with one heatmap per model (columns), shared y-axis (datasets).
    Each model gets its own vmin/vmax (not shared), and its own colorbar.
    """

    style = style or HeatmapStyle(
        hatch_map={2: "//", 3: "//"}
    )

    if model_order is None:
        model_order = list(model_to_cells_df.keys())

    # Prepare mats for each model (also establishes ds/art labels)
    prepared = {}
    for m in model_order:
        data, level_mat, ds_labels, art_labels = _prepare_heatmap_mats(
            model_to_cells_df[m],
            articles_order=articles_order,
            ds_order=ds_order,
            agg=agg,
            signif_mode=signif_mode,
            thresholds=thresholds,
        )
        prepared[m] = (data, level_mat, ds_labels, art_labels)

    # shared ds/art labels (assumes same datasets + articles across models)
    _, _, ds_labels, art_labels = prepared[model_order[0]]
    n_rows, n_cols = len(ds_labels), len(art_labels)
    n_models = len(model_order)

    # figure sizing: width ~ models * cols * cell_size, height ~ rows * cell_size
    fig_w = n_models * n_cols * cell_size + 0.2
    fig_h = n_rows * cell_size + 0.6
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(fig_w, fig_h),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    if n_models == 1:
        axes = [axes]

    for k, m in enumerate(model_order):
        ax = axes[k]
        data, level_mat, _, _ = prepared[m]

        # per-model scale
        absmax = np.nanmax(np.abs(data))
        if not np.isfinite(absmax) or absmax == 0:
            absmax = 1.0
        vmin, vmax = -absmax, absmax

        im = ax.imshow(data, aspect="equal", vmin=vmin, vmax=vmax, cmap=style.cmap)
        #ax.set_aspect("equal", adjustable="box")

        # ticks / labels
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(art_labels, rotation=45, ha="right", fontsize=12)

        ax.set_yticks(np.arange(n_rows))
        if k == 0:
            ax.set_yticklabels(
                [f"{pretty_dataset_mapping.get(ds, ds)}$({article_mapping.get(ds, ds)})$" for ds in ds_labels],
                fontsize=15
            )
            ax.set_ylabel("Dataset", fontsize=18)
        else:
            ax.tick_params(axis="y", labelleft=False)

        # grid
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        # set x tick label to fontsize 15
        ax.set_xticklabels(art_labels, rotation=45, ha="right", fontsize=15)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=style.grid_linewidth)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xlabel("Article", fontsize=18)
        ax.set_title(m, fontsize=25)

        # overlays
        _add_sig_stars(ax, level_mat, fontsize=7.0)

        exp = (model_to_expected_cells or {}).get(m, [])
        hatch = (model_to_hatch_cells or {}).get(m, [])
        local = (model_to_local_cells or {}).get(m, [])
        train = (model_to_train_datasets or {}).get(m, None)

        _add_expected_direction_boxes(
            ax,
            expected_cells=exp,
            ds_labels=ds_labels,
            art_labels=art_labels,
            train_datasets=train,
        )
        colors = {l: EXPECTED_EDGE_COLORS['increase'] if l[1] == target_article else EXPECTED_EDGE_COLORS['decrease'] for l in hatch}
        _add_hatching_cells(ax, hatch, ds_labels, art_labels, hatch="///", colors=colors)

        colors = {l: EXPECTED_EDGE_COLORS['increase'] if l[1] == target_article else EXPECTED_EDGE_COLORS['decrease'] for l in local}
        #_add_local_triangles(ax, local, ds_labels, art_labels, colors=colors, alpha=0.25)
        _add_hatching_cells(ax, local, ds_labels, art_labels, hatch="\\\\\\", colors=colors)
        #_add_local_rectangles(ax, local, ds_labels, art_labels, color="black", alpha=0.25)

        # colorbar
        if cbar_mode == "inside":
            cax = inset_axes(ax, width="4.5%", height="70%", loc="center right", borderpad=0.7)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        elif cbar_mode == "right":
            cbar = fig.colorbar(
                im, ax=ax,
                fraction=cbar_frac_right,
                pad=cbar_pad_right,
                aspect=cbar_aspect_right,
                orientation="vertical",
            )
        elif cbar_mode == "below":
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "bottom",
                size="3%",
                pad=0.80,
            )

            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

            fig.canvas.draw()

            # sicher: Tick-Parameter auf der X-Achse setzen
            cbar.ax.xaxis.set_tick_params(labelsize=10)

            # extra-sicher: Ticklabels neu setzen (überschreibt alles)
            labels = [t.get_text() for t in cbar.ax.get_xticklabels()]
            cbar.ax.set_xticklabels(labels, fontsize=10)

            cbar.ax.xaxis.set_ticks_position("bottom")
            cbar.ax.xaxis.set_label_position("bottom")
            #cbar.ax.tick_params(labelsize=5)
            for t in cbar.ax.get_xticklabels():
                t.set_fontsize(4)
        else:
            raise ValueError("cbar_mode must be 'inside', 'right', or 'below'")

        cbar.set_label(r"$\Delta \mathbb{P}(\mathrm{article})$", fontsize=12)
        cbar.ax.tick_params(labelsize=15, length=2.0, width=0.8, direction="in", pad=4.0)
        cbar.formatter = FuncFormatter(lambda x, pos: f"{x:.1f}\%")
        cbar.update_ticks()

    # one highlight legend for the whole figure (place on left-most axis)
    if show_legend:
        add_highlight_legend_fig(
            fig,
        )

    if title:
        fig.suptitle(title, fontsize=10, y=1.02)

    #fig.tight_layout()
    return fig, axes



def run_multi_model_heatmaps_like_your_main(
    *,
    base_model_ids: Sequence[str],
    all_interesting_classes: dict,
    articles_order: Sequence[str] = ("der", "die", "das", "den", "dem", "des"),
    results_decoder_dir: str = "results/decoder",
    out_dir: Optional[str] = None,
    signif_mode: str = "fdr",
    thresholds: Tuple[float, float, float] = (0.05, 0.01, 0.001),
    accuracy_threshold: float = 0.99,
    rule_axis: str = "gender",                 # kept for parity; not used directly here
    restrict_expected_to_config: bool = False, # kept for parity; not used directly here
    verbose: bool = True,
    skip_on_error: bool = True,
):
    """
    Creates ONE compact multi-panel heatmap per (config, steer article->other_article),
    with ALL base models shown next to each other, shared y-axis (datasets),
    and per-panel color scale (different magnitudes allowed).

    Output file naming mirrors your previous function, but grouped across models.

    Requirements:
      - plot_article_delta_heatmap_multi_models(...) exists (the multi-panel plotter)
      - analyze_steering_all_articles_from_files(...) exists
      - GradiendGenderCaseConfiguration, statistical_analysis_config_classes, article_mapping, init_matplotlib, IMAGE_DIR exist
      - EXPECTED_EDGE_COLORS and add_highlight_legend etc. are already defined like in your script
    """
    import os
    import matplotlib.pyplot as plt

    init_matplotlib(use_tex=True)

    if out_dir is None:
        models_tag = "__".join([m.replace("/", "-") for m in base_model_ids])
        out_dir = f"{IMAGE_DIR}/decoder_heatmaps_multi/{models_tag}"
    os.makedirs(out_dir, exist_ok=True)

    # Iterate the same way as before over your config class groups
    for articles, config_keys in statistical_analysis_config_classes.items():
        # For each config key, we will make one combined figure per steering direction
        for config_key in config_keys:
            # Reference config (datasets/articles) from the first model
            ref_cfg = GradiendGenderCaseConfiguration(*config_key, model_id=base_model_ids[0])
            ref_datasets = list(ref_cfg.datasets)
            ref_articles = list(ref_cfg.articles)

            # Your old logic assumes exactly 2 articles in config.articles
            # If more exist, we still follow your "pick the other" behavior per target.
            for article in ref_articles:
                other_article = [a for a in ref_articles if a != article]
                if not other_article:
                    continue
                other_article = other_article[0]

                # Determine TG (this_dataset) once from reference config datasets
                this_dataset = [ds for ds in ref_datasets if article_mapping.get(ds) == article]
                if len(this_dataset) == 0:
                    if verbose:
                        print(f"[WARN] No TG datasets found for config={ref_cfg.id} article={article}")
                    continue

                # Collect per-model results
                model_to_cells: Dict[str, pd.DataFrame] = {}
                model_to_expected: Dict[str, List[Tuple[str, str, str]]] = {}
                model_to_hatch: Dict[str, List[Tuple[str, str]]] = {}
                model_to_other_hatch: Dict[str, List[Tuple[str, str]]] = {}
                model_to_local: Dict[str, List[Tuple[str, str]]] = {}
                model_to_train: Dict[str, Iterable[str]] = {}

                for base_model_id in base_model_ids:
                    cfg = GradiendGenderCaseConfiguration(*config_key, model_id=base_model_id)

                    try:
                        model_path = cfg.gradiend_dir
                    except Exception as e:
                        if verbose:
                            print(f"[ERROR] Could not get model path for {base_model_id} config={cfg.id}: {e}")
                        if skip_on_error:
                            continue
                        else:
                            raise

                    try:
                        cells_df, gf, alpha = analyze_steering_all_articles_from_files(
                            base_eval_csv=f"{results_decoder_dir}/{base_model_id}_base_eval.csv",
                            mod_eval_csv=f"{model_path}_decoder_sig_analysis_mod_eval.csv",
                            base_neutral_csv=f"{results_decoder_dir}/{base_model_id}_base_neutral.csv",
                            mod_neutral_csv=f"{model_path}_decoder_sig_analysis_mod_neutral.csv",
                            select_token=other_article,
                            TG=this_dataset,
                            articles_to_score=list(articles_order),
                            accuracy_threshold=accuracy_threshold,
                        )
                        if cells_df is None or cells_df.empty:
                            raise ValueError("cells_df empty")

                        model_to_cells[base_model_id] = cells_df

                        # Highlights (can vary by model; safe to store per model)
                        model_to_expected[base_model_id] = cfg.abstract_rule_based_gender_case_article_pairs[article]
                        model_to_hatch[base_model_id] = cfg.memorization_gender_case_article_pairs[article]
                        model_to_local[base_model_id] = cfg.local_rule_based_gender_case_article_pairs
                        model_to_train[base_model_id] = cfg.datasets

                    except Exception as e:
                        if verbose:
                            print(f"[ERROR] {base_model_id} | {ref_cfg.id} | {article}->{other_article}: {e}")
                        if skip_on_error:
                            continue
                        else:
                            raise

                if len(model_to_cells) == 0:
                    if verbose:
                        print(f"[SKIP] No models produced data for {ref_cfg.id} | {article}->{other_article}")
                    continue

                # Ensure a shared ds_order across all included models
                ds_union = list(article_mapping.keys())

                # Build a multi-model figure
                fig_title = rf"{ref_cfg.id} | steer {article}->{other_article}"

                # Small tweak: pass model_order only for successful models
                ok_models = list(model_to_cells.keys())

                # If you want per-panel titles with alpha/gf, update the plotter slightly:
                # - replace ax.set_title(m, ...) with ax.set_title(model_to_title.get(m, m), ...)
                #
                # Here we do it with a tiny monkey-patch approach: just pass "model_order"
                # and then adjust titles after plotting.
                fig, axes = plot_article_delta_heatmap_multi_models(
                    model_to_cells_df=model_to_cells,
                    articles_order=list(articles_order),
                    ds_order=ds_union,
                    agg="mean",
                    signif_mode=signif_mode,
                    thresholds=thresholds,
                    style=None,
                    model_to_expected_cells=model_to_expected,
                    model_to_hatch_cells=model_to_hatch,
                    model_to_other_hatch_cells=model_to_other_hatch,
                    model_to_local_cells=model_to_local,
                    model_to_train_datasets=model_to_train,
                    model_order=ok_models,
                    #title=fig_title,
                    cell_size=0.42,
                    cbar_mode='below',
                    show_legend=True,
                    target_article=other_article,
                )

                # Overwrite subplot titles to include alpha/gf (optional but useful)
                for ax, m in zip(axes, ok_models):
                    ax.set_title(pretty_model_mapping[m], fontsize=18)

                out_path = os.path.join(out_dir, f"{ref_cfg.id}__{article}_to_{other_article}__MULTI.pdf")
                fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
                plt.show()
                plt.close(fig)

                if verbose:
                    print(f"Saved: {out_path}")


if __name__ == "__main__":
    models = [
        'bert-base-german-cased',
        'gbert-large',
        'ModernGBERT_1B',
        'EuroBERT-210m',
        'german-gpt2',
        'Llama-3.2-3B',
    ]
    run_multi_model_heatmaps_like_your_main(base_model_ids=models, all_interesting_classes=all_interesting_classes)

    #for model in models:
    #    run_heatmap_like_your_main(base_model_id=model, all_interesting_classes=all_interesting_classes)
