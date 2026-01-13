import os
import re
import json
import time
import hashlib
import pickle
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from gradiend.evaluation.xai.io import (
    all_interesting_classes,
    control_config,
    load_model_with_gradiends_by_configs,
)
from gradiend.util import init_matplotlib


# -----------------------
# Persistent cache helpers
# -----------------------

def _slug(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s[:200] if len(s) > 200 else s


def _stable_hash_dict(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _load_pickle(path: Path):
    if not path.exists():
        return None
    try:
        return pickle.loads(path.read_bytes())
    except Exception:
        return None


def _save_pickle(path: Path, obj) -> None:
    _atomic_write_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def _default_cache_dir() -> Path:
    return Path(os.environ.get("MAMUT_CACHE_DIR", ".cache/mamut_intersections"))


def _config_hash(config) -> str:
    try:
        if isinstance(config, dict):
            return _stable_hash_dict(config)
        return _stable_hash_dict({"config": config})
    except Exception:
        return hashlib.sha256(repr(config).encode("utf-8")).hexdigest()[:16]


def _length_cache_path(cache_dir: Path, model_id: str, scope: str, map_to_neuron: bool) -> Path:
    # Length may differ by scope/map_to_neuron depending on your Gradient container.
    # If it does not, you can drop scope/map_to_neuron from this key.
    key = _stable_hash_dict(
        {"model_id": model_id, "scope": scope, "map_to_neuron": bool(map_to_neuron), "v": 1}
    )
    return cache_dir / "meta" / f"length_{_slug(key)}.pkl"


def _proportions_cache_path(
    cache_dir: Path,
    *,
    model_id: str,
    article_id: str,
    scope: str,
    map_to_neuron: bool,
    pretty_model_names: list[str],
    length: int,
    cfg_hash: str,
) -> Path:
    payload = {
        "model_id": model_id,
        "article_id": article_id,
        "scope": scope,
        "map_to_neuron": bool(map_to_neuron),
        "pretty_model_names": list(pretty_model_names),
        "length": int(length),
        "config_hash": cfg_hash,
        "schema_v": 2,  # proportions-based cache schema
    }
    key = _stable_hash_dict(payload)
    return cache_dir / "proportions" / f"{_slug(key)}.pkl"


# -----------------------
# Main function (improved cached)
# -----------------------

def plot_intersection_proportions(
    config,
    model_id: str = "bert-base-german-cased",
    scope: str = "neuron",
    map_to_neuron: bool = True,
    cache_dir: str = None,
    force_recompute: bool = False,
):
    """
    Caching strategy (fast plotting, minimal disk):
    1) Cache LENGTH per (model_id, scope, map_to_neuron) to avoid loading gradients just to get len().
    2) Cache final PLOT DATA (topks + proportions per subset label) per (config, model_id, scope, map_to_neuron, length, pretty model names).
       -> When present, plotting is just load+plot (<10s).
    3) Optional restart safety while computing: we write results incrementally after each subset.

    Note: We do NOT cache topk_sets anymore (too big).
    """

    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()

    # ---- compute article_id early (cheap) ----
    # We need names -> requires loading. But you already have names only from load_model_with_gradiends.
    # So we keep this later, and use a provisional id for the length cache only (which doesn't need names).

    # ---- length cache (avoid expensive load if possible) ----
    len_path = _length_cache_path(cache_dir, model_id=model_id, scope=scope, map_to_neuron=map_to_neuron)
    length = None if force_recompute else _load_pickle(len_path)
    if isinstance(length, dict) and "length" in length:
        length = int(length["length"])

    gradients = None
    names = None
    pretty_ids = None

    if length is None:
        # Need to load once to determine length
        gradients, names, pretty_ids = load_model_with_gradiends_by_configs(
            config,
            device=torch.device("cpu"),
            model_id=model_id,
            return_pretty_model_id=True,
        )
        # Derive pretty model names + length
        gradients = {n: g for n, g in zip(pretty_ids.values(), gradients.values())}
        model_names = list(gradients.keys())
        if len(model_names) < 2:
            raise ValueError("Need at least two models.")
        length = len(gradients[model_names[0]])

        _save_pickle(len_path, {"model_id": model_id, "scope": scope, "map_to_neuron": bool(map_to_neuron), "length": int(length), "t": time.time()})
    else:
        # We still need gradients to compute proportions if not cached; defer loading until needed.
        pass

    # topks (log-spaced + length)
    topks = (10 ** np.arange(1, int(np.log10(length)) + 1))
    topks = topks[topks <= length]
    if length not in topks:
        topks = np.append(topks, length)
    topks = tuple(topks.astype(int))

    cfg_hash = _config_hash(config)

    # If we didn't load gradients above, we still don't know article_id nor pretty model names.
    # However, proportions cache key depends on them -> must load minimally.
    if gradients is None:
        gradients, names, pretty_ids = load_model_with_gradiends_by_configs(
            config,
            device=torch.device("cpu"),
            model_id=model_id,
            return_pretty_model_id=True,
        )
        gradients = {n: g for n, g in zip(pretty_ids.values(), gradients.values())}

    model_names = list(gradients.keys())
    if len(model_names) < 2:
        raise ValueError("Need at least two models.")

    article_id = "_".join(
        set([n.split(": ")[-1].replace("<", "").replace(">", "") for n in names.values()])
    )

    output = f"img/intersection_subsets_{model_id}_{article_id}_{scope}.pdf"

    # ---- proportions cache (fast path) ----
    prop_path = _proportions_cache_path(
        cache_dir,
        model_id=model_id,
        article_id=article_id,
        scope=scope,
        map_to_neuron=map_to_neuron,
        pretty_model_names=model_names,
        length=length,
        cfg_hash=cfg_hash,
    )

    cache = None if force_recompute else _load_pickle(prop_path)
    if cache and cache.get("complete") is True:
        # FAST PLOT (no heavy computation)
        topks_cached = tuple(cache["topks"])
        series = cache["series"]  # label -> proportions list
        _plot_series(topks_cached, series, output=output)
        return

    # ---- compute (restart-safe) ----
    # Cache layout (small):
    # {
    #   "meta": {...},
    #   "topks": [...],
    #   "series": { label: [p1, p2, ...] },
    #   "complete": bool
    # }
    if not cache:
        cache = {
            "meta": {
                "created": time.time(),
                "model_id": model_id,
                "article_id": article_id,
                "scope": scope,
                "map_to_neuron": bool(map_to_neuron),
                "pretty_model_names": list(model_names),
                "length": int(length),
                "config_hash": cfg_hash,
                "schema_v": 2,
            },
            "topks": list(map(int, topks)),
            "series": {},
            "complete": False,
        }
    else:
        # If topks changed, we need to recompute (proportions depend on exact k list).
        # If you want partial reuse across topks changes, keep a "raw intersections" cache,
        # but that's back to large data. Here we enforce topks consistency.
        if tuple(cache.get("topks", [])) != tuple(map(int, topks)):
            cache = {
                "meta": cache.get("meta", {}),
                "topks": list(map(int, topks)),
                "series": {},
                "complete": False,
            }
        if "series" not in cache:
            cache["series"] = {}

    # Precompute top-k lists per model for all k ONCE? Too big.
    # Instead: for each k, get per-model top-k and compute intersections for all subsets incrementally.
    # This keeps memory bounded: per k we hold sets for each model, then discard.
    #
    # We compute intersection LENGTHS per subset across k, then normalize into proportions.
    # We store final proportions per subset label into cache["series"].

    subsets = []
    for r in [2]:
        subsets.extend(itertools.combinations(model_names, r))

    # If restarting, skip already computed labels
    done_labels = set(cache["series"].keys())

    # For neuron-scope normalization, you previously used max(intersection_lengths) per subset
    # (over k). That requires seeing all k lengths first.
    # We'll compute all intersection lengths per subset (small: #topks * #subsets ints),
    # then derive proportions and store them.
    #
    # This is still small: e.g., 6 models => subsets of size 2..5:
    # C(6,2)+C(6,3)+C(6,4)+C(6,5)=57; topks maybe ~6-8; 57*8=456 ints.
    intersection_lengths_by_label: dict[str, list[int]] = {}

    # If we are resuming and had already computed some series, we don't have their raw lengths.
    # We'll recompute only missing labels from scratch (still ok), but we can do it efficiently:
    # compute lengths for all missing labels in one pass over k.
    missing_subsets = []
    missing_labels = []
    for subset in subsets:
        label = " ∩ ".join(subset)
        if label not in done_labels:
            missing_subsets.append(subset)
            missing_labels.append(label)

    if missing_subsets:
        # initialize arrays
        for label in missing_labels:
            intersection_lengths_by_label[label] = []

        for k in topks:
            # compute top-k sets for each model at this k
            sets_for_k = {}
            for name in model_names:
                topk = gradients[name].get_top_k_neurons(
                    top_k=int(k),
                    scope=scope,
                    map_to_neuron=map_to_neuron,
                )
                sets_for_k[name] = set(map(int, topk))

            # now compute intersections for missing subsets only
            for subset, label in zip(missing_subsets, missing_labels):
                inter = set.intersection(*(sets_for_k[m] for m in subset))
                intersection_lengths_by_label[label].append(len(inter))

        # convert to proportions and write each label (restart-safe)
        for subset, label in zip(missing_subsets, missing_labels):
            lens = intersection_lengths_by_label[label]
            if scope == "neuron":
                denom = max(lens) if lens else 1
                props = [l / denom if denom else 0.0 for l in lens]
            else:
                props = [l / kk for l, kk in zip(lens, topks)]
            cache["series"][label] = props

            # write after each label so restarts keep progress
            _save_pickle(prop_path, cache)

    # mark complete and save once
    cache["complete"] = True
    _save_pickle(prop_path, cache)

    # Plot from cached series (fast)
    _plot_series(topks, cache["series"], output=output)


def _plot_series(topks, series: dict[str, list[float]], output: str):
    plt.figure(figsize=(9, 6))
    for label, proportions in series.items():
        plt.plot(topks, proportions, marker="o", label=label)

    plt.xlabel("top-k")
    plt.ylabel("Proportion")
    plt.grid(True)
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.show()


# -----------------------
# Usage stays the same
# -----------------------
if __name__ == "__main__":
    init_matplotlib(use_tex=True)

    plot_fnc = plot_intersection_proportions

    models = [
        "bert-base-german-cased",
        "gbert-large",
        "german-gpt2",
        "EuroBERT-210m",
        "ModernGBERT_1B",
        "Llama-3.2-3B",
    ]

    scope = "weight"
    map_to_neuron = False

    for model_id in models:
        for articles, config in all_interesting_classes.items():
            print(articles)
            try:
                plot_fnc(config, model_id=model_id, scope=scope, map_to_neuron=map_to_neuron)
            except Exception as e:
                print(f"Error for {articles} with model {model_id}:")
                print(e)

        try:
            plot_fnc(list(control_config.values())[0], model_id=model_id, scope=scope, map_to_neuron=map_to_neuron)
        except Exception as e:
            print(e)
