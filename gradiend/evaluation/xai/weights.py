import itertools

import torch

from gradiend.model import GradiendModel

types_mfn_old = [
    'MF',
    'MN',
    'FN',
]

types_case_old = [
    'AD',
    'GD',
    'NA',
    'ND',
    'NG',
    'DA',
]

types_mfn = [
    'N_MF',
    'A_MF',
    'D_MF',
    'G_MF',
    'N_MN',
    'A_MN',
    'N_FN',
    'A_FN',
    'D_FN',
    'G_FN',
]

types_case = [
    'AD_M',
    'AD_F',
    'AD_N',
    'GD_M',
    'GD_N',
    'NA_M',
    'ND_M',
    'ND_F',
    'ND_N',
    'NG_M',
    'NG_F',
    'NG_N',
    'DA_M',
    'DA_F',
    'DA_N',
]

def load_weights(base_dir='results/experiments/gradiend', type='mfn', mode='weight'):
    if type == 'case':
        type_ids = types_case
    elif type == 'mfn':
        type_ids = types_mfn
    else:
        raise ValueError("Type must be 'mfn' or 'case', got: {type}")

    weight_mapping = {}
    for type in type_ids:
        gradiend = GradiendModel.from_pretrained(f'{base_dir}/{type}_neutral_augmented/dim_1_inv_gradient/bert-base-german-cased/0')

        weights = gradiend.decoder[0].weight.data.squeeze()
        #weights = gradiend.decoder[0].bias.data.squeeze()
        #weights = gradiend.encoder[0].weight

        if mode == 'weight':
            weight_mapping[type] = weights.data.squeeze()
        else:
            weight_mapping[type] = gradiend.transform_weights_to_neurons(part='decoder')

    return weight_mapping


def load_case_weights(base_dir, mode='weight'):
    if mode == 'neuron':
        raise NotImplementedError("Neuron mode not implemented for case weights.")

    weight_mapping = {}
    for type in ['AD', 'GD', 'NA', 'ND', 'NG', 'DA']:
        gradiend = GradiendModel.from_pretrained(f'{base_dir}/{type}/dim_1_inv_gradient/bert-base-german-cased/1')
        #weights = gradiend.decoder[0].weight.data.squeeze()
        weights = gradiend.encoder[0].weight
        weight_mapping[type] = weights.data.squeeze()
    return weight_mapping


import torch
import numpy as np
from typing import Dict, Iterable, List, Union

def get_topk_weight_indices_for_gradiends(model_with_gradiends, top_k, scope='neuron'):
    top_k_weights = set()
    intersection_weights = set()
    intersection_indices = set()
    union_weights = set()
    union_indices = set()
    for model_with_gradiend in model_with_gradiends.values():
        if scope == 'weight':
            indices = model_with_gradiend.get_top_k_neurons(top_k=top_k, map_to_neuron=False, scope=scope, return_weight_indices=True)
            indices = set(indices)
        else:
            neuron_ids, indices = model_with_gradiend.get_top_k_neurons(top_k=top_k, map_to_neuron=True, scope=scope, return_weight_indices=True)
            indices = set.union(*[set(v) for v in indices.values()])
        union_indices = union_indices.union(indices)
        if not intersection_weights:
            #intersection_weights = set(mg_top_k_weights.tolist())
            intersection_indices = indices
        else:
            #intersection_weights = intersection_weights.intersection(set(mg_top_k_weights.tolist()))
            intersection_indices = intersection_indices.intersection(indices)

    return intersection_indices, union_indices

def get_topk_weight_indices(top_k: int, return_sorted: bool = True, type='mfn', mode='weight'):
    """
    Compute top-k weight indices per type and their intersection across all types.

    Args:
        weight_mapping: dict mapping type_name -> 1D or 2D weight vector (torch.Tensor or np.ndarray).
        top_k: number of top neurons to select per type.
        return_sorted: whether to sort indices by descending absolute importance.

    Returns:
        result: dict mapping each type -> list of top-k indices
        common_topk: list of neuron indices appearing in all top-k sets (intersection)
    """
    assert mode in {'weight', 'neuron'}

    result = {}
    weight_mapping = load_weights(type=type, mode=mode)
    for type_name, w in weight_mapping.items():
        # Convert to torch tensor on CPU
        if isinstance(w, np.ndarray):
            w_t = torch.from_numpy(w)
        elif isinstance(w, torch.Tensor):
            w_t = w.cpu()
        else:
            raise TypeError(f"Invalid type for {type_name}: {type(w)}")

        # Aggregate over classes if needed
        if w_t.dim() == 2:
            importance = w_t.abs().sum(dim=0)
        elif w_t.dim() == 1:
            importance = w_t.abs()
        else:
            raise ValueError(f"Unexpected shape for {type_name}: {w_t.shape}")

        # Limit top_k to available neurons
        k = min(top_k, importance.numel())

        # Get top-k indices
        _, top_idx = torch.topk(importance, k=k, largest=True, sorted=return_sorted)
        result[type_name] = top_idx.tolist()

    # Compute intersection (common neurons)
    all_sets = [set(v) for v in result.values()]
    common_topk = list(set.intersection(*all_sets)) if all_sets else []

    print(f"Common top-{top_k} weights across all types: {len(common_topk)} found")

    return result, common_topk






def check_weight_relation_mfn(base_dir):
    weight_mapping = load_weights(base_dir)

    print('mean abs MF weights:', weight_mapping['MF'].abs().mean().item())
    print('mean abs MN weights:', weight_mapping['MN'].abs().mean().item())
    print('mean abs FN weights:', weight_mapping['FN'].abs().mean().item())

    aggregated = weight_mapping['MF'] - weight_mapping['MN'] + weight_mapping['FN']
    print('mean abs aggregated weights:', aggregated.abs().mean().item())

import torch

def check_weight_relation_mfn_topk(base_dir, top_k=100, normalize=False):
    """
    Compute aggregated weights using top-k neurons per type, augmenting others with 0.

    Args:
        base_dir (str): directory containing saved weights.
        top_k (int): number of top neurons per type (by absolute weight) to consider.
    """
    weight_mapping = load_weights(base_dir)
    num_neurons = weight_mapping['MF'].numel()

    # Function to get top-k neuron mask
    def topk_mask(weights, k):
        topk_idx = torch.topk(weights.abs(), k).indices
        mask = torch.zeros_like(weights)
        mask[topk_idx] = weights[topk_idx]
        return mask

    # Create augmented vectors (top-k preserved, others set to 0)
    mf_aug = topk_mask(weight_mapping['MF'], top_k)
    mn_aug = topk_mask(weight_mapping['MN'], top_k)
    fn_aug = topk_mask(weight_mapping['FN'], top_k)

    if normalize:
        max_mf = mf_aug.abs().max().item()
        max_mn = mn_aug.abs().max().item()
        max_fn = fn_aug.abs().max().item()

        mf_aug /= max_mf if max_mf != 0 else 1.0
        mn_aug /= max_mn if max_mn != 0 else 1.0
        fn_aug /= max_fn if max_fn != 0 else 1.0

    print(f'Mean abs MF weights (top-{top_k}):', mf_aug.abs().mean().item())
    print(f'Mean abs MN weights (top-{top_k}):', mn_aug.abs().mean().item())
    print(f'Mean abs FN weights (top-{top_k}):', fn_aug.abs().mean().item())

    # Aggregated weight vector
    aggregated = mf_aug - mn_aug + fn_aug
    print(f'Mean abs aggregated weights (top-{top_k}):', aggregated.abs().mean().item())

import torch

def check_weight_relation_mfn_topk_intersection(base_dir, top_k=1000):
    """
    Compute aggregated weights using only neurons that are in the top-k for all three types.

    Args:
        base_dir (str): directory containing saved weights.
        top_k (int): number of top neurons per type (by absolute weight) to consider.
    """
    weight_mapping = load_weights(base_dir)

    # Get top-k indices for each type
    mf_topk_idx = torch.topk(weight_mapping['MF'].abs(), top_k).indices
    mn_topk_idx = torch.topk(weight_mapping['MN'].abs(), top_k).indices
    fn_topk_idx = torch.topk(weight_mapping['FN'].abs(), top_k).indices

    # Find intersection
    intersection_idx = torch.tensor(
        list(set(mf_topk_idx.tolist()) & set(mn_topk_idx.tolist()) & set(fn_topk_idx.tolist())),
        dtype=torch.long
    )

    if len(intersection_idx) == 0:
        print("No overlapping neurons in top-k across MF, MN, FN.")
        return

    # Restrict weights to overlapping neurons
    mf_overlap = weight_mapping['MF'][intersection_idx]
    mn_overlap = weight_mapping['MN'][intersection_idx]
    fn_overlap = weight_mapping['FN'][intersection_idx]

    print(f'Number of overlapping neurons: {len(intersection_idx)}')
    print('Mean abs MF weights (overlap):', mf_overlap.abs().mean().item())
    print('Mean abs MN weights (overlap):', mn_overlap.abs().mean().item())
    print('Mean abs FN weights (overlap):', fn_overlap.abs().mean().item())

    # Aggregated weight vector
    aggregated = mf_overlap - mn_overlap + fn_overlap
    print('Mean abs aggregated weights (overlap):', aggregated.abs().mean().item())
    aggregated_abs = mf_overlap.abs() + mn_overlap.abs() + fn_overlap.abs()
    print('Mean abs aggregated absolute weights (overlap):', aggregated_abs.mean().item())


import torch

def check_weight_relation_mfn_topk_normalized(base_dir, top_k=10000):
    """
    Compute normalized aggregated weights using top-k neurons and show approximate directional cancellation.
    """
    weight_mapping = load_weights(base_dir)

    if top_k == 'all':
        mf_topk_idx = torch.arange(weight_mapping['MF'].numel())
        mn_topk_idx = torch.arange(weight_mapping['MN'].numel())
        fn_topk_idx = torch.arange(weight_mapping['FN'].numel())
    else:
        # Get top-k indices for each type
        mf_topk_idx = torch.topk(weight_mapping['MF'].abs(), top_k).indices
        mn_topk_idx = torch.topk(weight_mapping['MN'].abs(), top_k).indices
        fn_topk_idx = torch.topk(weight_mapping['FN'].abs(), top_k).indices

    # Intersection of top-k neurons
    intersection_idx = torch.tensor(
        list(set(mf_topk_idx.tolist()) & set(mn_topk_idx.tolist()) & set(fn_topk_idx.tolist())),
        dtype=torch.long
    )

    if len(intersection_idx) == 0:
        print("No overlapping neurons in top-k across MF, MN, FN.")
        return

    # Extract overlapping weights
    mf_overlap = weight_mapping['MF'][intersection_idx]
    mn_overlap = weight_mapping['MN'][intersection_idx]
    fn_overlap = weight_mapping['FN'][intersection_idx]

    # Normalize each vector to unit L2 norm
    mf_norm = mf_overlap / mf_overlap.norm(p=2)
    mn_norm = mn_overlap / mn_overlap.norm(p=2)
    fn_norm = fn_overlap / fn_overlap.norm(p=2)

    # Aggregated vector
    aggregated = mf_norm - mn_norm + fn_norm

    print(f'Number of overlapping neurons: {len(intersection_idx)}')
    print('Mean abs MF (normalized):', mf_norm.abs().mean().item())
    print('Mean abs MN (normalized):', mn_norm.abs().mean().item())
    print('Mean abs FN (normalized):', fn_norm.abs().mean().item())
    print('Mean abs aggregated (normalized):', aggregated.abs().mean().item())
    print('L2 norm of aggregated vector:', aggregated.norm().item())


import torch
import matplotlib.pyplot as plt

def check_most_important_neurons(base_dir):
    weight_mapping = load_weights(base_dir)

    # Print top 10 neurons per type
    for type, weights in weight_mapping.items():
        sorted_weights, indices = torch.sort(weights.abs(), descending=True)
        print(f'Top 10 important neurons for {type}:')
        for i in range(10):
            print(f'Neuron {indices[i].item()} with weight {sorted_weights[i].item():.4f}')
        print()

    # Prepare data for plotting
    top_k_values = [m * 10**e for e in range(0, 8) for m in [1, 2, 5]]
    mf_mn_overlap = []
    mf_fn_overlap = []
    mn_fn_overlap = []
    all_overlap = []

    for top_k in top_k_values:
        mf_topk = set(torch.topk(weight_mapping['MF'].abs(), top_k).indices.tolist())
        mn_topk = set(torch.topk(weight_mapping['MN'].abs(), top_k).indices.tolist())
        fn_topk = set(torch.topk(weight_mapping['FN'].abs(), top_k).indices.tolist())

        intersection_mf_mn = mf_topk.intersection(mn_topk)
        intersection_mf_fn = mf_topk.intersection(fn_topk)
        intersection_mn_fn = mn_topk.intersection(fn_topk)
        intersection_all = intersection_mf_mn.intersection(fn_topk)

        # Store percentages
        mf_mn_overlap.append(len(intersection_mf_mn) / top_k * 100)
        mf_fn_overlap.append(len(intersection_mf_fn) / top_k * 100)
        mn_fn_overlap.append(len(intersection_mn_fn) / top_k * 100)
        all_overlap.append(len(intersection_all) / top_k * 100)

        print(f'Intersection of top {top_k} neurons:')
        print(f'MF and MN: {len(intersection_mf_mn)} neurons ({mf_mn_overlap[-1]:.2f}%)')
        print(f'MF and FN: {len(intersection_mf_fn)} neurons ({mf_fn_overlap[-1]:.2f}%)')
        print(f'MN and FN: {len(intersection_mn_fn)} neurons ({mn_fn_overlap[-1]:.2f}%)')
        print(f'All three: {len(intersection_all)} neurons ({all_overlap[-1]:.2f}%)')
        print()

    # Plot the overlaps
    plt.figure(figsize=(8, 6))
    plt.plot(top_k_values, mf_mn_overlap, marker='o', label='MF ∩ MN')
    plt.plot(top_k_values, mf_fn_overlap, marker='s', label='MF ∩ FN')
    plt.plot(top_k_values, mn_fn_overlap, marker='^', label='MN ∩ FN')
    plt.plot(top_k_values, all_overlap, marker='D', label='MF ∩ MN ∩ FN')

    plt.xscale('log')
    plt.xlabel('Top-k Neurons (log scale)', fontsize=12)
    plt.ylabel('Intersection Percentage (%)', fontsize=12)
    plt.title('Overlap of Most Important Neurons Across Categories', fontsize=13)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
import torch
import itertools
import matplotlib.pyplot as plt

def check_casewise_neuron_overlap(base_dir, mode='all', alpha_pairs=0.4):
    """
    Analyze and plot neuron overlap consistency across grammatical case contrasts.

    Args:
        base_dir (str): Path to the base directory containing saved weight files.
        mode (str): Either 'all' (intersection across all contrasts per case)
                    or 'pairwise' (show all pairwise overlaps + per-case mean).
        alpha_pairs (float): Opacity for individual pairwise curves.
    """
    weight_mapping = load_case_weights(base_dir)

    # Define contrasts per grammatical case
    case_contrasts = {
        'N': ['NG', 'ND', 'NA'],
        'G': ['NG', 'GD', 'GA'],
        'D': ['ND', 'GD', 'DA'],
        'A': ['NA', 'GA', 'DA']
    }

    top_k_values = [m * 10**e for e in range(0, 8) for m in [1, 2, 5]]
    overlap_data = {case: [] for case in case_contrasts}
    pairwise_data = {case: {} for case in case_contrasts}  # store pairwise overlaps if needed

    for top_k in top_k_values:
        for case, contrasts in case_contrasts.items():
            # Ensure contrasts exist
            valid_contrasts = [c for c in contrasts if c in weight_mapping]
            if len(valid_contrasts) < 2:
                overlap_data[case].append(float('nan'))
                continue

            # Compute top-k sets
            topk_sets = {
                c: set(torch.topk(weight_mapping[c].abs(), top_k).indices.tolist())
                for c in valid_contrasts
            }

            if mode == 'all':
                # Intersection across all contrasts for this case
                intersection_all = set.intersection(*topk_sets.values())
                overlap_percent = len(intersection_all) / top_k * 100
                overlap_data[case].append(overlap_percent)
                print(f"Top {top_k:6d} — Case {case}: {overlap_percent:6.2f}% (ALL)")

            elif mode == 'pairwise':
                # Compute each pair separately
                overlaps = {}
                for (c1, c2) in itertools.combinations(valid_contrasts, 2):
                    inter = topk_sets[c1].intersection(topk_sets[c2])
                    percent = len(inter) / top_k * 100
                    overlaps[(c1, c2)] = percent
                    print(f"Top {top_k:6d} — Case {case}: {c1} ∩ {c2} = {percent:6.2f}%")

                # Save each pair's overlap
                for pair, val in overlaps.items():
                    pairwise_data[case].setdefault(pair, []).append(val)

                # Average across pairs for mean trend line
                overlap_data[case].append(sum(overlaps.values()) / len(overlaps))

            else:
                raise ValueError("mode must be 'all' or 'pairwise'")

    # ---- Plot ----
    plt.figure(figsize=(9, 6))

    if mode == 'all':
        # Plot just one line per case
        for case, values in overlap_data.items():
            plt.plot(top_k_values, values, marker='o', label=f'{case}-case')
    else:
        # Plot individual pairwise lines (faint)
        for case, pair_dict in pairwise_data.items():
            for (c1, c2), values in pair_dict.items():
                plt.plot(top_k_values, values, '--', alpha=alpha_pairs, label=f'{case}: {c1}∩{c2}')

        # Plot bold per-case average trend
        for case, values in overlap_data.items():
            plt.plot(top_k_values, values, marker='o', linewidth=2.5, label=f'{case}-case (avg)')

    plt.xscale('log')
    plt.xlabel('Top-k Neurons (log scale)', fontsize=12)
    plt.ylabel('Intersection Percentage (%)', fontsize=12)
    title_suffix = "All Contrasts" if mode == 'all' else "Pairwise Contrasts"
    plt.title(f'Overlap of Most Important Neurons Across {title_suffix}', fontsize=13)
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()




import torch
import matplotlib.pyplot as plt

def analyze_neuron_specificity(base_dir, top_k=1000):
    weight_mapping = load_weights(base_dir)

    # Get top-k neurons for each type
    mf_topk = set(torch.topk(weight_mapping['MF'].abs(), top_k).indices.tolist())
    mn_topk = set(torch.topk(weight_mapping['MN'].abs(), top_k).indices.tolist())
    fn_topk = set(torch.topk(weight_mapping['FN'].abs(), top_k).indices.tolist())

    # Exclusive sets
    m_only = (mf_topk.intersection(mn_topk)) - fn_topk
    f_only = (mf_topk.intersection(fn_topk)) - mn_topk
    n_only = (mn_topk.intersection(fn_topk)) - mf_topk

    # Alternatively, if you mean "only appears in one set at all"
    only_mf = mf_topk - mn_topk - fn_topk
    only_mn = mn_topk - mf_topk - fn_topk
    only_fn = fn_topk - mf_topk - mn_topk

    print(f'For top {top_k} neurons:')
    print(f'M-only neurons: {len(m_only)} ({len(m_only)/top_k*100:.2f}%)')
    print(f'F-only neurons: {len(f_only)} ({len(f_only)/top_k*100:.2f}%)')
    print(f'N-only neurons: {len(n_only)} ({len(n_only)/top_k*100:.2f}%)')
    print()
    print(f'Only-in-one-type neurons:')
    print(f'Only in MF: {len(only_mf)} ({len(only_mf)/top_k*100:.2f}%)')
    print(f'Only in MN: {len(only_mn)} ({len(only_mn)/top_k*100:.2f}%)')
    print(f'Only in FN: {len(only_fn)} ({len(only_fn)/top_k*100:.2f}%)')

    # Optional: visualize exclusive neurons
    labels = ['M-only', 'F-only', 'N-only']
    values = [len(m_only), len(f_only), len(n_only)]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(f'Exclusive Neurons at Top-{top_k}')
    plt.ylabel('Number of Neurons')
    plt.tight_layout()
    plt.show()

    return {
        'M-only': m_only,
        'F-only': f_only,
        'N-only': n_only,
        'Only_MF': only_mf,
        'Only_MN': only_mn,
        'Only_FN': only_fn
    }


import torch
import matplotlib.pyplot as plt

def analyze_directional_consistency(base_dir):
    weight_mapping = load_weights(base_dir)
    mf, mn, fn = weight_mapping['MF'], weight_mapping['MN'], weight_mapping['FN']

    top_k_values = [m * 9**e for e in range(0, 8) for m in [1, 2, 5]]

    n_weights = mf.shape[0]
    top_k_values.append(n_weights)
    same_dir_mf_mn, same_dir_mf_fn, same_dir_mn_fn = [], [], []

    for top_k in top_k_values:
        if top_k == n_weights:
            mf_topk = set(range(n_weights))
            mn_topk = set(range(n_weights))
            fn_topk = set(range(n_weights))
        else:
            mf_topk = torch.topk(mf.abs(), top_k).indices.tolist()
            mn_topk = torch.topk(mn.abs(), top_k).indices.tolist()
            fn_topk = torch.topk(fn.abs(), top_k).indices.tolist()

        # find neurons that are in both top-k sets
        mf_mn_common = set(mf_topk) & set(mn_topk)
        mf_fn_common = set(mf_topk) & set(fn_topk)
        mn_fn_common = set(mn_topk) & set(fn_topk)

        def same_direction_ratio(common_set, w1, w2, invert=False):
            if not common_set:
                return 0.0
            signs1 = torch.sign(w1[list(common_set)])
            signs2 = torch.sign(w2[list(common_set)] * (-1 if invert else 1))
            same = torch.sum(signs1 == signs2).item()
            return same / len(common_set) * 100

        same_dir_mf_mn.append(same_direction_ratio(mf_mn_common, mf, mn))
        same_dir_mf_fn.append(same_direction_ratio(mf_fn_common, mf, fn, invert=True))
        same_dir_mn_fn.append(same_direction_ratio(mn_fn_common, mn, fn))

        print(f"Top {top_k}:")
        print(f"  MF–MN same direction: {same_dir_mf_mn[-1]:.2f}%")
        print(f"  MF–FN same direction: {same_dir_mf_fn[-1]:.2f}%")
        print(f"  MN–FN same direction: {same_dir_mn_fn[-1]:.2f}%")
        print()

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(top_k_values, same_dir_mf_mn, marker='o', label='MF–MN')
    plt.plot(top_k_values, same_dir_mf_fn, marker='s', label='MF–FN')
    plt.plot(top_k_values, same_dir_mn_fn, marker='^', label='MN–FN')

    plt.xscale('log')
    plt.xlabel('Top-k Neurons (log scale)', fontsize=12)
    plt.ylabel('Same-direction percentage (%)', fontsize=12)
    plt.title('Directional Consistency Across Top-k Weight Intersections', fontsize=13)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('img/directional_consistency_mfn.png', dpi=400)
    plt.show()


def analyze_case_directional_consistency(base_dir):
    weight_mapping = load_case_weights(base_dir)

    # Expected keys: 'NG', 'ND', 'NA', 'GD', 'GA', 'DA'
    available = list(weight_mapping.keys())
    print(f"Available contrasts: {available}")

    top_k_values = [m * 9**e for e in range(0, 8) for m in [1, 2, 5]]
    consistency_results = {pair: [] for pair in itertools.combinations(available, 2)}

    def same_direction_ratio(common, w1, w2):
        if not common:
            return 0.0
        s1, s2 = torch.sign(w1[list(common)]), torch.sign(w2[list(common)])
        same = torch.sum(s1 == s2).item()
        return same / len(common) * 100

    for top_k in top_k_values:
        topk_sets = {name: set(torch.topk(weights.abs(), top_k).indices.tolist())
                     for name, weights in weight_mapping.items()}

        for (k1, k2) in itertools.combinations(available, 2):
            common = topk_sets[k1].intersection(topk_sets[k2])
            ratio = same_direction_ratio(common, weight_mapping[k1], weight_mapping[k2])
            consistency_results[(k1, k2)].append(ratio)

    # Plot
    plt.figure(figsize=(9, 6))
    for (k1, k2), values in consistency_results.items():
        plt.plot(top_k_values, values, marker='o', label=f'{k1}–{k2}')

    plt.xscale('log')
    plt.xlabel('Top-k Neurons (log scale)', fontsize=12)
    plt.ylabel('Same-direction percentage (%)', fontsize=12)
    plt.title('Directional Consistency Across Case Contrasts', fontsize=13)
    plt.legend(ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

import torch
import matplotlib.pyplot as plt

def analyze_casewise_directional_consistency(base_dir):
    weight_mapping = load_case_weights(base_dir)

    # Define which contrasts involve each case
    case_contrasts = {
        'N': ['NG', 'ND', 'NA'],
        'G': ['NG', 'GD', 'GA'],
        'D': ['ND', 'GD', 'DA'],
        'A': ['NA', 'GA', 'DA']
    }

    top_k_values = [m * 9**e for e in range(0, 8) for m in [1, 2, 5]]
    results = {case: [] for case in case_contrasts}

    for top_k in top_k_values:
        for case, contrasts in case_contrasts.items():
            # Ensure all required contrasts exist in your data
            contrasts = [c for c in contrasts if c in weight_mapping]
            if len(contrasts) < 2:
                results[case].append(float('nan'))
                continue

            # Compute intersection of top-k neurons across all contrasts for this case
            topk_sets = [
                set(torch.topk(weight_mapping[c].abs(), top_k).indices.tolist())
                for c in contrasts
            ]
            common = set.intersection(*topk_sets)
            if not common:
                results[case].append(0.0)
                continue

            # Extract weights for all contrasts
            weights = torch.stack([weight_mapping[c][list(common)] for c in contrasts])
            signs = torch.sign(weights)

            # Consistency: all signs same (either all +1 or all -1)
            consistent = torch.sum(torch.all(signs == signs[0, :], dim=0)).item()
            ratio = consistent / len(common) * 100
            results[case].append(ratio)

            print(f"Case {case} | Top {top_k} | Common neurons: {len(common)} | "
                  f"Consistent: {ratio:.2f}%")

    # Plot
    plt.figure(figsize=(8, 6))
    for case, values in results.items():
        plt.plot(top_k_values, values, marker='o', label=f'{case}-case')

    plt.xscale('log')
    plt.xlabel('Top-k Neurons (log scale)', fontsize=12)
    plt.ylabel('Same-direction percentage (%)', fontsize=12)
    plt.title('Directional Consistency for Each Case Across Contrasts', fontsize=13)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return results


import torch


def test_vector_difference_assumption(base_dir, top_k=None):
    """
    Test the assumption w_XY ~ v_X - v_Y for X,Y in {M,F,N}.
    Optionally restrict to top-k neurons per type.
    """
    weight_mapping = load_weights(base_dir)

    # Optionally select top-k neurons per type
    def select_topk(weights, k):
        if k is None:
            return weights
        topk_idx = torch.topk(weights.abs(), k).indices
        vec = torch.zeros_like(weights)
        vec[topk_idx] = weights[topk_idx]
        return vec

    MF = select_topk(weight_mapping['MF'], top_k)
    MN = select_topk(weight_mapping['MN'], top_k)
    FN = select_topk(weight_mapping['FN'], top_k)

    # Stack pairwise weight vectors into a matrix: each column is one neuron vector
    # Each row corresponds to a pair: MF, MN, FN
    W = torch.stack([MF, MN, FN], dim=0)  # shape: (3, num_neurons)

    # We want to find v_M, v_F, v_N s.t.
    # MF ≈ v_M - v_F
    # MN ≈ v_M - v_N
    # FN ≈ v_F - v_N

    # Setup linear system: W = A V, where V = [v_M; v_F; v_N]
    # For each neuron, we have 3 equations
    # A is 3x3:
    # MF row: [1, -1, 0]
    # MN row: [1, 0, -1]
    # FN row: [0, 1, -1]

    num_neurons = W.shape[1]
    A = torch.tensor([
        [1, -1, 0],
        [1, 0, -1],
        [0, 1, -1]
    ], dtype=torch.float32)  # shape: (3,3)

    residuals = []
    V_solutions = []

    for i in range(num_neurons):
        w = W[:, i].unsqueeze(1)  # shape: (3,1)
        device = w.device  # get the device of the weight
        A_device = A.to(device)  # move A to same device

        # Solve least squares using new API
        result = torch.linalg.lstsq(A_device, w)
        v = result.solution  # shape: (3,1)
        V_solutions.append(v.squeeze())

        # Compute residual norm
        residual = torch.norm(A_device @ v - w).item()
        residuals.append(residual)

    V_solutions = torch.stack(V_solutions, dim=1)  # shape: (3, num_neurons)
    residuals = torch.tensor(residuals)

    print(f'Average residual per neuron: {residuals.mean().item():.6f}')
    print(f'Max residual per neuron: {residuals.max().item():.6f}')
    print(f'Fraction of neurons with residual < 0.1: {(residuals < 0.1).float().mean().item():.2%}')

    # save

    return V_solutions, residuals


if __name__ == '__main__':
    base_dir = 'results/experiments/gradiend'
    #check_weight_relation_mfn(base_dir)
    check_most_important_neurons(base_dir)
    #analyze_directional_consistency(base_dir)
    #analyze_case_directional_consistency('results/experiments/gradiend')
    #analyze_casewise_directional_consistency('results/experiments/gradiend')
    #check_casewise_neuron_overlap(base_dir, mode='pairwise')
    #check_weight_relation_mfn_topk('results/experiments/gradiend', top_k=1000)
    #check_weight_relation_mfn_topk_intersection('results/experiments/gradiend')
    #check_weight_relation_mfn_topk_normalized('results/experiments/gradiend')
    #test_vector_difference_assumption(base_dir)