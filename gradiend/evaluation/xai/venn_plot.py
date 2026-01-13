from matplotlib.patches import Patch
from matplotlib_venn import venn3, venn2
import matplotlib.colors as mcolors
from tqdm import tqdm

from gradiend.evaluation.xai.io import all_interesting_classes, control_config, load_model_with_gradiends_by_configs

from venn import venn

from gradiend.util import init_matplotlib

scope = 'neuron'
scope = 'weight'

map_to_neuron = scope == 'neuron'

def create_venn_diagram_3(gradiends, names, top_k=1000, class_colors=None, output=None):
    assert len(gradiends) == 3, "Venn diagram only supported for exactly 3 models"
    assert len(names) == 3

    # Map names to gradients
    gradiends = {name.replace('_neutral_augmented', ''): grad for name, grad in zip(names.values(), gradiends.values())}
    model_names = list(gradiends.keys())

    # Get top-k neuron sets
    top_k_sets = {
        name: set(grad.get_top_k_neurons(top_k=top_k, scope=scope, map_to_neuron=map_to_neuron))
        for name, grad in gradiends.items()
    }

    A, B, C = model_names
    setA, setB, setC = top_k_sets[A], top_k_sets[B], top_k_sets[C]

    # Compute all 7 regions
    only_A = len(setA - setB - setC)
    only_B = len(setB - setA - setC)
    only_C = len(setC - setA - setB)

    AB = len((setA & setB) - setC)
    AC = len((setA & setC) - setB)
    BC = len((setB & setC) - setA)

    ABC = len(setA & setB & setC)


    plt.figure(figsize=(6, 6))
    v = venn3(
        subsets=(only_A, only_B, AB, only_C, AC, BC, ABC),
        set_labels=(A, B, C),
        #set_colors=(class_colors[A], class_colors[B], class_colors[C]),
        alpha=0.5,
    )

    subset_label_fontsize = 30 if top_k < 5000 else 25
    label_fontsize = 40 if top_k < 5000 else 30

    for txt in v.subset_labels:
        if txt:
            txt.set_fontsize(subset_label_fontsize)
            txt.set_fontweight("bold")
            #txt.set_path_effects([patheffects.withStroke(linewidth=1, foreground="white")])

    for txt in v.set_labels:
        if txt:
            txt.set_fontsize(label_fontsize)
            # set bold
            txt.set_fontweight("bold")

    for patch in v.patches:
        if patch:
            #patch.set_edgecolor("black")
            patch.set_linewidth(3.0)



def create_venn_diagram_2(gradiends, names, top_k=1000, class_colors=None, output=None):
    """
    Create a 2-set Venn diagram for top-k neurons of two gradient models.

    Arguments:
        gradiends: dict-like, e.g. {"modelA": gradA, "modelB": gradB}
        names: dict-like, same key order as gradiends, e.g. {"A": "modelA", "B": "modelB"}
        top_k: number of top neurons to consider
        class_colors: optional dict {"modelA": colorA, "modelB": colorB}
    """

    assert len(gradiends) == 2, "This function only supports exactly 2 models."
    assert len(names) == 2, "Names must contain exactly 2 entries."

    # Map logical names to gradient objects, identical logic to 3-set function
    gradiends = {name: grad for name, grad in zip(names.values(), gradiends.values())}
    model_names = list(gradiends.keys())  # deterministic

    A, B = model_names

    # Compute top-k sets
    top_k_sets = {
        name: set(grad.get_top_k_neurons(top_k=top_k, scope=scope, map_to_neuron=map_to_neuron))
        for name, grad in gradiends.items()
    }

    setA, setB = top_k_sets[A], top_k_sets[B]

    # Compute regions
    only_A = len(setA - setB)
    only_B = len(setB - setA)
    AB = len(setA & setB)

    # Default color handling
    if class_colors is None:
        class_colors = {
            A: "red",
            B: "blue",
        }

    plt.figure(figsize=(6, 6))
    v = venn2(
        subsets=(only_A, only_B, AB),
        set_labels=(A, B),
        #set_colors=(class_colors[A], class_colors[B]),
        alpha=0.5,
    )

    subset_label_fontsize = 30 if top_k < 5000 else 25
    label_fontsize = 40 if top_k < 5000 else 30

    for txt in v.subset_labels:
        if txt:
            txt.set_fontsize(subset_label_fontsize + 10)
            txt.set_fontweight("bold")

    for txt in v.set_labels:
        if txt:
            txt.set_fontsize(label_fontsize + 10)
            # set bold
            txt.set_fontweight("bold")

    for patch in v.patches:
        if patch:
            patch.set_linewidth(3.0)

def plot_for_gradiend_config(config, top_k=1000, model_id='bert-base-german-cased', id=''):
    gradiends, names, pretty_ids = load_model_with_gradiends_by_configs(config, device=torch.device('cpu'), model_id=model_id, return_pretty_model_id=True)
    gradiends = {n: g for n, g in zip(pretty_ids.values(), gradiends.values())}
    articles = {pretty_ids[k]: v.split(': ')[1].split('-') for k, v in names.items()}
    model_names = list(gradiends.keys())
    gradiend_lengths = {name: len(grad) for name, grad in gradiends.items()}
    if len(set(gradiend_lengths.values())) != 1:
        raise ValueError(f"Gradiends have different lengths: {gradiend_lengths}")

    article_id = "_".join(set([name.split(': ')[-1].replace('<', '').replace('>', '') for name in names.values()]))
    output = f'img/venn/{model_id}_{article_id}_scope_{scope}.pdf'
    os.makedirs('img/venn', exist_ok=True)

    # Suppose you have 4 or 6 models:
    if len(model_names) < 2:
        raise ValueError("At least two models are required for Venn diagram.")
    elif len(model_names) == 2:
        # use normal library
        create_venn_diagram_2(
            gradiends,
            pretty_ids,
            top_k=top_k,
            output=output,
        )
    elif len(model_names) == 3:
        # use normal library
        create_venn_diagram_3(
            gradiends,
            pretty_ids,
            top_k=top_k,
            output=output,
        )
    else:
        assert len(model_names) in (4, 5, 6)
        sets = {
            name: set(gradiends[name].get_top_k_neurons(top_k=top_k, map_to_neuron=map_to_neuron, scope=scope))
            for name in model_names
        }

        if len(model_names) == 4:
            renamed_sets = {}
            for k, v in sets.items():
                k_articles = articles[k]
                k_articles = r"\!\to\!".join(k_articles)
                new_key = rf'{k}: ${k_articles}$'
                renamed_sets[new_key] = v
            sets = renamed_sets
        ax = plt.gca()
        v = venn(sets,
             ax=ax,
             legend_loc="upper center",
             #legend_ncol=2,
             )

        # move legend upward
        #plt.legend(
        #    loc="upper center",
        #    #bbox_to_anchor=(0.5, 0.5),  # (x, y) — increase y to move further up
        #    ncol=2,  # optional: spread across columns
        #    frameon=False  # optional: cleaner look
        #)

        # Subset labels
        for txt in ax.texts:
            txt.set_fontweight("bold")
            txt.set_color("black")
            subset_label_fontsize = 20 if top_k < 5000 else 15
            label_fontsize = 30 if top_k < 5000 else 20

            if txt.get_fontsize() < 20:
                txt.set_fontsize(subset_label_fontsize)  # subset labels
            else:
                txt.set_fontsize(label_fontsize)  # set labels

    #for patch in ax.patches:
        #patch.set_edgecolor("black")
       # patch.set_linewidth(3.0)

    # Legend
    ax = plt.gca()
    if len(model_names) <= 3:
        ax = plt.gca()
        ax.legend(
            #loc="center left",
            #bbox_to_anchor=(1.2, 0.5),
            frameon=False
        )
    else:
        colors = []
        for i in range(len(model_names)):
            p = v.patches[i]
            colors.append(p.get_facecolor())

        def darken(color, factor=0.7):
            rgb = np.array(mcolors.to_rgb(color))
            return np.clip(rgb * factor, 0, 1)

        handles = [
            Patch(facecolor=c, edgecolor=darken(c), label=name)
            for c, name in zip(colors, model_names)
        ]

        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.94, 0.5),
            frameon=True,
            fontsize=20,
        )

    #plt.title(f"Top-{top_k} Inclusive Neuron Overlap")
    plt.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()

import itertools
import matplotlib.pyplot as plt

def plot_intersection_proportions(config,  model_id='bert-base-german-cased'):
    gradients, names, pretty_ids = load_model_with_gradiends_by_configs(config, device=torch.device('cpu'), model_id=model_id, return_pretty_model_id=True)
    gradients = {n: g for n, g in zip(pretty_ids.values(), gradients.values())}
    model_names = list(gradients.keys())

    if len(model_names) < 2:
        raise ValueError("Need at least two models.")

    length = len(gradients[model_names[0]])

    topks = (10 ** np.arange(1, int(np.log10(length)) + 1))
    topks = topks[topks <= length]
    # include length if not already
    if length not in topks:
        topks = np.append(topks, length)
    topks = tuple(topks.astype(int))

    # Prepare file name
    article_id = "_".join(
        set([n.split(': ')[-1].replace('<', '').replace('>', '') for n in names.values()])
    )
    output = f"img/intersection_subsets_{model_id}_{article_id}_{scope}.pdf"

    # Pre-load all top-k sets for speed
    topk_sets = {
        k: {
            name: set(
                gradients[name].get_top_k_neurons(
                    top_k=k, scope=scope, map_to_neuron=map_to_neuron
                )
            )
            for name in model_names
        }
        for k in topks
    }

    # Prepare plot
    plt.figure(figsize=(9, 6))

    # We compute proportions for every non-empty proper subset of models (size 2 to n)
    for r in range(2, len(model_names)):
        for subset in itertools.combinations(model_names, r):

            label = " ∩ ".join(subset)

            intersection_lengths = []
            for k in topks:
                sets_for_k = topk_sets[k]
                inter = set.intersection(*(sets_for_k[m] for m in subset))
                intersection_lengths.append(len(inter))

            if scope == 'neuron':
                number_of_neurons = max(intersection_lengths)
                proportions = [l / number_of_neurons for l in intersection_lengths]
            else:
                proportions = [l / k for l, k in zip(intersection_lengths, topks)]


            plt.plot(topks, proportions, marker="o", label=label)

    plt.xlabel("top-k")
    plt.ylabel("Proportion")
    plt.grid(True)
    #plt.legend(bbox_to_anchor=(0.05, 1), loc="upper left")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()



def plot_intersection_proportions_fast(config,  model_id='bert-base-german-cased', id=''):
    assert scope == 'weight'

    output = f"img/intersection_subsets/{model_id}_{id}_{scope}.pdf"
    cache_file = f'results/topk_sets_{model_id}_{scope}__{id}.pt'
    if os.path.exists(cache_file):
        top_ks_proportions, topks = torch.load(cache_file)
        print(f"Loaded top-k sets from cache: {cache_file}")
    else:
        gradients, names, pretty_ids = load_model_with_gradiends_by_configs(config, device=torch.device('cpu'), model_id=model_id, return_pretty_model_id=True)
        print(f'Loaded gradiends for models: {list(pretty_ids.values())}')
        gradients = {n: g for n, g in zip(pretty_ids.values(), gradients.values())}
        model_names = list(gradients.keys())

        if len(model_names) < 2:
            raise ValueError("Need at least two models.")

        length = len(gradients[model_names[0]])

        topks = (10 ** np.arange(1, int(np.log10(length)) + 1))
        topks = topks[topks <= length]
        # include length if not already
        if length not in topks:
            topks = np.append(topks, length)
        topks = tuple(topks.astype(int))
        print(f'Using top-k values: {topks}')

        # Pre-load all top-k sets for speed
        topk_sets = {}
        for name, gradiend in tqdm.tqdm(gradients.items(), desc="Computing ranked weights"):
            print('Computing ranked weights for model:', name)
            weights = gradiend.gradiend.decoder[0].weight.data.abs().cpu()
            ranked_weights = torch.argsort(weights.view(-1), descending=True).numpy()

            for k in topks:
                if k not in topk_sets:
                    topk_sets[k] = {}
                topk_sets[k][name] = set(ranked_weights[:k])


        # We compute proportions for every non-empty proper subset of models (size 2)
        print(f"Computing intersections for subsets of size 2")
        top_ks_proportions = {}
        for r in [2]: # range(2, len(model_names)):
            combs = itertools.combinations(model_names, r)
            for subset in tqdm(combs, desc="Computing subsets", total=len(list(combs))):

                label = " ∩ ".join(subset)

                intersection_lengths = []
                for k in topks:
                    sets_for_k = topk_sets[k]
                    inter = set.intersection(*(sets_for_k[m] for m in subset))
                    intersection_lengths.append(len(inter))

                if scope == 'neuron':
                    number_of_neurons = max(intersection_lengths)
                    proportions = [l / number_of_neurons for l in intersection_lengths]
                else:
                    proportions = [l / k for l, k in zip(intersection_lengths, topks)]

                top_ks_proportions[label] = proportions
        # save to cache
        torch.save((topk_sets, topks), cache_file)
        print(f"Saved top-k sets to cache: {cache_file}")

    print(f"Plotting intersection proportions to {output}")
    # Prepare plot
    plt.figure(figsize=(9, 4))
    for label, proportions in top_ks_proportions.items():
        plt.plot(topks, proportions, marker="x", label=label, zorder=3)

    plt.xlabel("top-k")
    plt.ylabel("Proportion")
    plt.grid(True, zorder=0)
    plt.legend(bbox_to_anchor=(0.05, 1), loc="upper left")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()
import os, itertools
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

def _compute_rank_from_weights_abs(weights_1d: torch.Tensor) -> torch.Tensor:
    """
    Build a rank array: rank[idx] = 1..N (1 = largest weight).
    weights_1d must be a 1D tensor on CPU.
    """
    # Sort indices by descending weight magnitude
    order = torch.argsort(weights_1d, descending=True)

    # rank[order[pos]] = pos+1
    n = weights_1d.numel()
    rank = torch.empty(n, dtype=torch.int64)
    rank[order] = torch.arange(1, n + 1, dtype=torch.int64)
    return rank


def _intersection_sizes_for_topks_from_ranks(ranks: list[torch.Tensor], topks: np.ndarray) -> np.ndarray:
    """
    Compute |∩_m Top_k(m)| for multiple k values using ranks.
    For a subset of models, define t[i] = max_m rank_m[i].
    Then intersection_size(k) = count(t <= k), which can be obtained from prefix sums of bincount(t).
    """
    # Stack ranks and take elementwise max across models in subset
    # (All ranks are 1..N)
    t = ranks[0].clone()
    for r in ranks[1:]:
        t = torch.maximum(t, r)

    n = t.numel()

    # Histogram of thresholds; minlength ensures indices up to n exist
    counts = torch.bincount(t, minlength=n + 1)  # counts[0] unused
    prefix = torch.cumsum(counts, dim=0)         # prefix[k] = number of elements with t <= k

    # Gather intersection sizes at desired k values
    k_tensor = torch.as_tensor(topks, dtype=torch.int64)
    sizes = prefix[k_tensor].cpu().numpy().astype(np.int64)
    return sizes


def plot_intersection_proportions_fast(config, model_id='bert-base-german-cased', id='', scope='weight'):
    assert scope == 'weight'

    output = f"img/intersection_subsets/{model_id}_{id}_{scope}.pdf"
    cache_file = f"results/intersection_prefix_{model_id}_{scope}__{id}.pt"

    if False and os.path.exists(cache_file):
        payload = torch.load(cache_file)
        top_ks_proportions = payload["top_ks_proportions"]
        topks = payload["topks"]
        print(f"Loaded cache: {cache_file}")
    else:
        gradients, names, pretty_ids = load_model_with_gradiends_by_configs(
            config, device=torch.device('cpu'), model_id=model_id, return_pretty_model_id=True
        )
        gradients = {n: g for n, g in zip(pretty_ids.values(), gradients.values())}
        model_names = list(gradients.keys())

        if len(model_names) < 2:
            raise ValueError("Need at least two models.")

        # Determine N (flattened weight size) from first model
        example = next(iter(gradients.values()))
        weights0 = example.gradiend.decoder[0].weight.data.abs().cpu().view(-1)
        n = weights0.numel()

        # Same top-k grid as before: powers of 10 plus full length
        decades = 10 ** np.arange(0, int(np.log10(n)) + 1)
        topks = np.concatenate([d * np.array([1, 2, 5]) for d in decades])
        topks = topks[(topks >= 1) & (topks <= n)].astype(np.int64)
        topks = topks[topks <= n]
        if n not in topks:
            topks = np.append(topks, n)
        topks = topks.astype(np.int64)

        print(f"N={n:,}")
        print(f"Using top-k values: {tuple(map(int, topks))}")

        # Build rank arrays for each model (exact, no top-k sets)
        ranks_by_model = {}
        for name, gr in tqdm.tqdm(gradients.items(), desc="Building rank arrays"):
            w = gr.gradiend.decoder[0].weight.data.abs().cpu().view(-1)
            ranks_by_model[name] = _compute_rank_from_weights_abs(w)

        # Compute intersections for subsets (here: size 2 as in your code)
        top_ks_proportions = {}
        pairs = list(itertools.combinations(model_names, 2))
        for a, b in tqdm.tqdm(pairs, desc="Computing pair intersections"):
            label = rf"$ {a} \cap {b} $"
            print("LABEL:", repr(label))

            inter_sizes = _intersection_sizes_for_topks_from_ranks(
                [ranks_by_model[a], ranks_by_model[b]], topks
            )

            # Proportion = |intersection| / k
            proportions = inter_sizes / topks
            top_ks_proportions[label] = proportions

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save({"top_ks_proportions": top_ks_proportions, "topks": topks}, cache_file)
        print(f"Saved cache: {cache_file}")

    # Plot
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.figure(figsize=(9, 4))
    for label, proportions in top_ks_proportions.items():
        plt.plot(topks, proportions, marker="x", label=label, zorder=3)

    plt.xlabel("top-k")
    plt.ylabel("Proportion")
    plt.grid(True, zorder=0)
    plt.legend(bbox_to_anchor=(0.05, 1), loc="upper left")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()



if __name__ == "__main__":
    init_matplotlib(use_tex=True)
    plot_fnc = plot_for_gradiend_config

    models = [
        'bert-base-german-cased',
        'gbert-large',
        'german-gpt2',
        'EuroBERT-210m',
        'ModernGBERT_1B',
        'Llama-3.2-3B',
    ]
    topk = 1000

    for model_id in models:
        for articles, config in all_interesting_classes.items():
            print(articles)
            #continue
            try:
                plot_fnc(config, model_id=model_id, id='_'.join(sorted(articles)), top_k=topk)
            except Exception as e:
                print(f"Error for {articles} with model {model_id}:")
                print(e)

        try:
            plot_fnc(list(control_config.values())[0], model_id=model_id, id=list(control_config.keys())[0], top_k=topk)
        except Exception as e:
            print(e)

