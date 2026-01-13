import numpy as np
import itertools
import torch

scope = 'neuron' #oder 'weight'
scope = 'weight'
map_to_neuron = scope == 'neuron'

def _pairwise_overlap_props(sets_by_name, denom="min"):
    props = []
    names = list(sets_by_name.keys())
    for a, b in itertools.combinations(names, 2):
        A, B = sets_by_name[a], sets_by_name[b]
        inter = len(A & B)

        if denom == "min":
            d = min(len(A), len(B))
        elif denom == "union":
            d = len(A | B)  # Jaccard
        else:
            raise ValueError("denom must be 'min' or 'union'")

        props.append(0.0 if d == 0 else inter / d)
    return props

def max_pairwise_overlap_for_config(config, model_id, top_k=1000, denom="min", device=torch.device("cpu")):
    gradients, names, pretty_ids = load_model_with_gradiends_by_configs(
        config, device=device, model_id=model_id, return_pretty_model_id=True
    )
    gradients = {n: g for n, g in zip(pretty_ids.values(), gradients.values())}
    model_names = list(gradients.keys())

    if len(model_names) < 2:
        return np.nan

    sets_by_name = {
        name: set(
            gradients[name].get_top_k_neurons(
                top_k=top_k, scope=scope, map_to_neuron=map_to_neuron
            )
        )
        for name in model_names
    }

    props = _pairwise_overlap_props(sets_by_name, denom=denom)
    return float(np.max(props)) if props else np.nan

def article_group_label(names_dict):
    return "_".join(
        sorted(
            set(
                n.split(": ")[-1].replace("<", "").replace(">", "")
                for n in names_dict.values()
            )
        )
    )

def latex_max_overlap_table(
    model_rows,
    all_interesting_classes,
    control_key,
    top_k=1000,
    denom="min",
    float_fmt="{:.3f}",
    device=torch.device("cpu"),
):
    items = list(all_interesting_classes.items())
    non_control = [(k, v) for k, v in items if k != control_key]
    control = [(k, v) for k, v in items if k == control_key]
    assert len(control) == 1, "control_key not found or not unique"

    ordered = non_control + control

    col_labels = []
    for _, cfg in ordered:
        _, names, _ = load_model_with_gradiends_by_configs(cfg, device=device, model_id=model_rows[0][1], return_pretty_model_id=True)
        col_labels.append(article_group_label(names))

    # LaTeX header
    header_cols = "l" + "r" * len(col_labels)
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(fr"    \begin{{tabular}}{{{header_cols}}}\toprule")
    lines.append(r"       \textbf{Model} & " + " & ".join([fr"\rotatebox{{90}}{{{c}}}" for c in col_labels]) + r" \\ \midrule")

    # Rows
    for latex_name, model_id in model_rows:
        vals = []
        for _, cfg in ordered:
            mx = max_pairwise_overlap_for_config(cfg, model_id=model_id, top_k=top_k, denom=denom, device=device)
            vals.append("--" if np.isnan(mx) else float_fmt.format(mx))
        lines.append("        " + latex_name + " & " + " & ".join(vals) + r" \\")
    lines.append(r"         \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \caption{Caption}")
    lines.append(r"    \label{tab:placeholder}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

if __name__ == '__main__':
    # ---------------- Example usage ----------------
    model_rows = [
       (r"\bert", "bert-base-german-cased"),
       (r"\gbert", "gbert-large"),
       (r"\modernbert", "ModernGBERT_1B"),
       (r"\eurobert", "EuroBERT-210m"),
       (r"\gpttwo", "german-gpt2"),
       (r"\llama", "Llama-3.2-3B"),
    ]

    from gradiend.evaluation.xai.io import all_interesting_classes, load_model_with_gradiends_by_configs, control_config

    config = all_interesting_classes | control_config

    control_key = "der_die_das_den"
    print(latex_max_overlap_table(model_rows, config, control_key, top_k=1000, denom="min"))
