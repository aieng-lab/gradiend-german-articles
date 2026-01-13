from gradiend.evaluation.xai.io import interesting_config_classes, control_config, semi_interesting_classes, \
    pretty_model_mapping, GradiendGenderCaseConfiguration, latex_article_mapping
from gradiend.util import init_matplotlib
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _pretty_config_id(cfg_key) -> str:
    """Return a stable, file-name friendly id for a config key."""
    if isinstance(cfg_key, (tuple, list)):
        return '$' + r"\!\to\!".join(map(str, sorted(cfg_key))) + '$'
    return str(cfg_key)


def _cache_config_id(cfg_key) -> str:
    """Return a stable, file-name friendly id for a config key."""
    if isinstance(cfg_key, (tuple, list)):
        return "_".join(map(str, sorted(cfg_key)))
    return str(cfg_key)




def _cache_path(model_id: str, scope: str, cfg_id: str) -> str:
    """Cache filename convention used by plot_intersection_proportions_fast."""
    return f"results/intersection_prefix_{model_id}_{scope}__{cfg_id}.pt"


def _collect_all_labels_from_caches(cfg_items, base_models, scope: str):
    """Return the union of all pair-labels found across all caches."""
    all_labels = set()
    for cfg_key, _ in cfg_items:
        cfg_id = _cache_config_id(cfg_key)
        for model_id in base_models:
            cache_file = _cache_path(model_id=model_id, scope=scope, cfg_id=cfg_id)
            if not os.path.exists(cache_file):
                continue
            payload = torch.load(cache_file)
            all_labels.update(payload["top_ks_proportions"].keys())
    return sorted(all_labels)


def _make_label_to_color(labels):
    """
    Create a stable label->color mapping for the whole figure.
    Uses Matplotlib's default cycle first, then falls back to a qualitative colormap if needed.
    """
    default_cycle = plt.rcParams.get("axes.prop_cycle", None)
    cycle_colors = []
    if default_cycle is not None:
        cycle_colors = [d.get("color") for d in default_cycle]
        cycle_colors = [c for c in cycle_colors if c is not None]

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8']


    label_to_color = {}
    label_to_marker = {}
    for i, lab in enumerate(labels):
        if i < len(cycle_colors):
            label_to_color[lab] = cycle_colors[i]
            label_to_marker[lab] = markers[i % len(markers)]

    if len(labels) > len(cycle_colors):
        cmap = plt.get_cmap("tab20")
        for i, lab in enumerate(labels[len(cycle_colors):], start=len(cycle_colors)):
            t = 0.0 if len(labels) == 1 else i / max(1, (len(labels) - 1))
            label_to_color[lab] = cmap(t)
            label_to_marker[lab] = markers[len(label_to_marker) % len(markers)]

    return label_to_color, label_to_marker


def plot_intersection_proportions_from_cache(
    configs_dict,
    base_models,
    scope="weight",
    mode="grid",          # "grid" or "per_model"
    swap_axes=True,
    output="img/intersection_subsets/intersection_cached.pdf",
    figsize_per_ax=(4.2, 2.6),
    max_pairs=None,
    legend_ncol=6,
    legend_y=0.99,
    legend_fontsize=10,
    alpha=0.9,
    linewidth=1.0,
    marker="o",
    min_topk=100,
    special_topk=1000,  # <-- add
    special_color="red",  # optional
    special_ls="-",  # optional
    special_lw=1.2,  # optional
    special_alpha=0.4,  # optional
):
    assert scope == "weight"
    assert mode in {"grid", "per_model"}

    cfg_items = list(configs_dict.items())
    n_cfg = len(cfg_items)
    n_models = len(base_models)

    all_labels = _collect_all_labels_from_caches(cfg_items, base_models, scope=scope)
    if max_pairs is not None:
        all_labels = all_labels[:max_pairs]
    all_labels_renamed = {}
    for l in all_labels:
        ds1, ds2 = l.strip('$').strip().split('\cap')
        ds1, ds2 = ds1.strip(), ds2.strip()
        #ds1_key, ds2_key = ds1.removeprefix(r'\grad'), ds2.removeprefix(r'\data')
        new_label = f'${ds1}\!\\leftrightarrow\!{ds2}$'
        all_labels_renamed[l] = new_label
    label_to_color, label_to_marker = _make_label_to_color(all_labels)


    linestyles = ["-", "--", ":", "-."]

    def _filter_topks(topks: np.ndarray, y: np.ndarray):
        """Filter (topks, y) by min_topk while preserving alignment."""
        if min_topk is None:
            return topks, y
        mask = np.asarray(topks) >= int(min_topk)
        return np.asarray(topks)[mask], np.asarray(y)[mask]

    def _draw_special(ax):
        if special_topk is None:
            return
        ax.axvline(
            special_topk,
            color=special_color,
            linestyle=special_ls,
            linewidth=special_lw,
            alpha=special_alpha,
            zorder=5,
        )



    if mode == "grid":
        # Decide grid shape
        if not swap_axes:
            n_rows, n_cols = n_cfg, n_models
        else:
            n_rows, n_cols = n_models, n_cfg

        fig_w = figsize_per_ax[0] * n_cols
        fig_h = figsize_per_ax[1] * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, sharex=True, sharey=True)

        def ax_for(r_cfg, c_model):
            return axes[r_cfg][c_model] if not swap_axes else axes[c_model][r_cfg]

        for r_cfg, (cfg_key, _cfg_unused) in enumerate(cfg_items):
            cfg_id = _pretty_config_id(cfg_key)
            cache_id = _cache_config_id(cfg_key)
            for c_model, model_id in enumerate(base_models):
                ax = ax_for(r_cfg, c_model)
                cache_file = _cache_path(model_id=model_id, scope=scope, cfg_id=cache_id)

                if not os.path.exists(cache_file):
                    ax.text(0.5, 0.5, f"Missing cache\n{cfg_id}\n{model_id}", ha="center", va="center")
                    ax.set_axis_off()
                    continue

                payload = torch.load(cache_file)
                top_ks_proportions = payload["top_ks_proportions"]
                topks = np.asarray(payload["topks"])

                labels_sorted = [lab for lab in all_labels if lab in top_ks_proportions]
                if max_pairs is not None:
                    labels_sorted = labels_sorted[:max_pairs]

                for lab in labels_sorted:
                    y = np.asarray(top_ks_proportions[lab])
                    x_f, y_f = _filter_topks(topks, y)
                    if x_f.size == 0:
                        continue
                    ax.plot(x_f, y_f, marker=label_to_marker[lab], linewidth=linewidth, alpha=alpha, color=label_to_color[lab])

                ax.set_xscale("log")
                ax.grid(True, zorder=0, linewidth=0.6)
                _draw_special(ax)
                # Titles and axis labels depend on orientation
        if not swap_axes:
            for c, model_id in enumerate(base_models):
                pretty_model_id = pretty_model_mapping[model_id]
                axes[0][c].set_title(pretty_model_id, fontsize=16)
            for r, (cfg_key, _) in enumerate(cfg_items):
                cfg_id = _pretty_config_id(cfg_key)
                axes[r][0].set_ylabel(f"{cfg_id}\nProportion", fontsize=10)
            for c in range(n_cols):
                axes[n_rows - 1][c].set_xlabel("Top-$k$", fontsize=10)
        else:
            for c, (cfg_key, _) in enumerate(cfg_items):
                cfg_id = _pretty_config_id(cfg_key)
                axes[0][c].set_title(cfg_id, fontsize=10)
            for r, model_id in enumerate(base_models):
                pretty_model_id = pretty_model_mapping[model_id]
                axes[r][0].set_ylabel(f"{pretty_model_id}\nProportion", fontsize=14)
            for c in range(n_cols):
                axes[n_rows - 1][c].set_xlabel("Top-$k$", fontsize=12)

        fig.tight_layout(rect=(0, 0, 1, 0.88))

    else:
        # One subplot per model, stacked vertically (always)
        n_rows, n_cols = n_models, 1
        fig_w = figsize_per_ax[0]
        fig_h = figsize_per_ax[1] * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, sharex=True, sharey=True)
        axes = axes[:, 0]

        for i_model, model_id in enumerate(base_models):
            ax = axes[i_model]

            pretty_model_id = pretty_model_mapping[model_id]
            # Model label on the left (as y-label), not as title
            ax.set_ylabel(pretty_model_id, fontsize=14)

            ax.set_xscale("log")
            ax.grid(True, zorder=0, linewidth=0.6)
            _draw_special(ax)

            for i_cfg, (cfg_key, _cfg_unused) in enumerate(cfg_items):
                cfg_id = _pretty_config_id(cfg_key)
                cache_id = _cache_config_id(cfg_key)
                cache_file = _cache_path(model_id=model_id, scope=scope, cfg_id=cache_id)
                if not os.path.exists(cache_file):
                    continue

                payload = torch.load(cache_file)
                top_ks_proportions = payload["top_ks_proportions"]
                topks = np.asarray(payload["topks"])

                ls = linestyles[i_cfg % len(linestyles)]
                labels_sorted = [lab for lab in all_labels if lab in top_ks_proportions]
                if max_pairs is not None:
                    labels_sorted = labels_sorted[:max_pairs]

                for lab in labels_sorted:
                    y = np.asarray(top_ks_proportions[lab])
                    x_f, y_f = _filter_topks(topks, y)
                    if x_f.size == 0:
                        continue
                    ax.plot(
                        x_f, y_f,
                        linestyle=ls,
                        marker=label_to_marker[lab],
                        linewidth=linewidth,
                        alpha=alpha,
                        color=label_to_color[lab],
                    )

            # Only bottom subplot gets x-label; y is always proportion (shared)
            if i_model == n_rows - 1:
                ax.set_xlabel("Top-$k$", fontsize=9)

        # Global y-axis label (Proportion) for per_model mode
        fig.text(0.02, 0.5, "Proportion", va="center", rotation="vertical", fontsize=10)

        fig.tight_layout(rect=(0.06, 0, 1, 0.88))

    # One global legend (pair colors only)
    legend_handles = [
        Line2D([0], [0], color=label_to_color[lab], marker=label_to_marker[lab], linewidth=linewidth, label=pretty_lab)
        for lab, pretty_lab in all_labels_renamed.items()
    ]
    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=min(legend_ncol, max(1, len(legend_handles))),
        fontsize=legend_fontsize,
        frameon=True,
        bbox_to_anchor=(0.5, legend_y),
    )

    # set ylabel fontsize
    for ax in fig.get_axes():
        ax.yaxis.label.set_size(12)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved figure to", output)



# Example usage
if __name__ == "__main__":
    init_matplotlib(use_tex=True)

    base_models = [
        "bert-base-german-cased",
        "gbert-large",
        "EuroBERT-210m",
        "ModernGBERT_1B",
        "german-gpt2",
        "Llama-3.2-3B",
    ]

    grid_configs1 = dict(interesting_config_classes)
    grid_configs2 = dict(semi_interesting_classes)
    grid_configs3 = dict(control_config)


    plot_intersection_proportions_from_cache(
        configs_dict=grid_configs3,
        base_models=base_models,
        mode="per_model",
        output="img/intersection_subsets/intersection_config.pdf",
        figsize_per_ax=(5, 2.6),
        legend_ncol=2,
        legend_y=0.93,
    )

    plot_intersection_proportions_from_cache(
        configs_dict=grid_configs1,
        base_models=base_models,
        mode="grid",
        output="img/intersection_subsets/intersection_grid.pdf",
        swap_axes=True,
        legend_y=0.92,
        legend_fontsize=10,
    )

    plot_intersection_proportions_from_cache(
        configs_dict=grid_configs2,
        base_models=base_models,
        mode="per_model",
        output="img/intersection_subsets/intersection_per_model.pdf",
        figsize_per_ax=(5, 2.6),
        legend_ncol=2,
        legend_y=0.93
    )
