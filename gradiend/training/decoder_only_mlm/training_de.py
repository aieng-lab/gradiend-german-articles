import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from gradiend.data import read_article_ds
from gradiend.training import create_de_training_dataset
from gradiend.training.decoder_only_mlm.model import DecoderModelWithMLMHead
from gradiend.util import init_matplotlib

ENCODING_PREFIX = ""

import itertools
import time
from typing import List, Iterable, Optional
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer


def _all_nonempty_subsets(items: Iterable[str], max_size: Optional[int] = None):
    items = list(items)
    max_size = max_size or len(items)
    for r in range(1, max_size + 1):
        for c in itertools.combinations(items, r):
            yield list(c)


def _output_path_for_subset(base_model: str, subset: List[str]):
    model_id = base_model.split("/")[-1]
    return f'results/decoder-mlm-head-gender-de/{"-".join(sorted(subset))}/{model_id}'

# todo needed?
def test_all_subsets(
    *,
    base_model: str = "dbmdz/german-gpt2",
    combinations: Iterable[str],
    batch_size: int = 4,
    epochs: int = 10,
    lr: float = 1e-6,
    max_subset_size: Optional[int] = None,
    min_avg_f1_success: float = 0.70,
    min_coverage_ratio_success: float = 0.90,
    min_tokens_with_f1_ge: float = 0.5,
    verbose: bool = True,
    sample_k: Optional[int] = None,
    strategy: str = "random",   # "random" oder "diverse"
):
    """
    Testet Subsets von Kombinationen. Optional Sampling:
      sample_k=None  -> teste alle Subsets
      sample_k=N     -> teste nur N Subsets, ausgewählt per 'random' oder 'diverse'
    """

    combos = list(combinations)
    all_subsets = list(_all_nonempty_subsets(combos, max_subset_size))

    if sample_k is not None:
        import random

        if strategy == "random":
            subsets = random.sample(all_subsets, min(sample_k, len(all_subsets)))

        elif strategy == "diverse":
            scored = []
            for s in all_subsets:
                prefixes = {x[0] for x in s}           # Vielfalt der ersten Buchstaben
                score = (len(prefixes), -len(s))      # mehr Prefixe, kleineres Set bevorzugen
                scored.append((score, s))
            scored.sort(reverse=True)
            subsets = [s for (_, s) in scored[:sample_k]]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if verbose:
            print(f"[Subset Selection] -> {len(subsets)} von {len(all_subsets)} Subsets (strategy={strategy})")

    else:
        subsets = all_subsets
        if verbose:
            print(f"[Subset Selection] -> teste alle {len(subsets)} Subsets")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    token_perf = {}  # token -> list of f1 values across subsets

    for subset in subsets:
        try:
            start = time.time()
            if verbose:
                print(f"\n=== Subset {subset} (size={len(subset)}) ===")

            train(
                base_model=base_model,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                combinations=subset,
            )
        except Exception as e:
            print(e)
            continue

        out_path = _output_path_for_subset(base_model, subset)
        if verbose:
            print(f"Lade Modell aus: {out_path}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(out_path)
            model = DecoderModelWithMLMHead.from_pretrained(out_path)
            model.to(DEVICE)
            model.eval()
        except Exception as e:
            print(f"Fehler beim Laden für {subset}: {e}")
            continue

        metrics = evaluate_split(model, tokenizer, split="test", batch_size=batch_size, combinations=subset)

        # === Coverage ===
        try:
            datasets = [read_article_ds(article=c, split="test") for c in subset]
            data = pd.concat(datasets, ignore_index=True)
            data = data[data['token_count'] == 1].reset_index(drop=True)
            expected_labels = set(l.lower() for l in data['label'].tolist())
            expected_tokens_count = len(expected_labels)
        except:
            expected_tokens_count = len(metrics['per_token'])

        covered_tokens = len(metrics['per_token'])
        coverage_ratio = covered_tokens / max(1, expected_tokens_count)

        # === per-token learning stats ===
        per_token = metrics['per_token']
        f1_vals = [v.get('f1', 0.0) for v in per_token.values()] if per_token else []
        mean_per_token_f1 = float(np.mean(f1_vals)) if f1_vals else 0.0
        num_tokens_good = sum(1 for f in f1_vals if f >= min_tokens_with_f1_ge)

        # ✅ Add per-token F1 into global score collector
        for tok, stats in per_token.items():
            token_perf.setdefault(tok, []).append(stats["f1"])

        success = (
                (metrics['f1'] >= min_avg_f1_success and coverage_ratio >= min_coverage_ratio_success)
                or (num_tokens_good >= 0.5 * max(1, expected_tokens_count))
        )

        results.append({
            "subset": subset,
            "subset_key": "-".join(sorted(subset)),
            "size": len(subset),
            "expected_tokens": expected_tokens_count,
            "covered_tokens": covered_tokens,
            "coverage_ratio": coverage_ratio,
            "acc": metrics["acc"],
            "macro_prec": metrics["prec"],
            "macro_rec": metrics["rec"],
            "macro_f1": metrics["f1"],
            "mean_per_token_f1": mean_per_token_f1,
            "num_tokens_f1_ge_threshold": num_tokens_good,
            "success": success,
            "runtime_s": time.time() - start,
        })

        del model
        torch.cuda.empty_cache()

    summary_df = pd.DataFrame(results).sort_values(["macro_f1", "coverage_ratio"], ascending=False).reset_index(
        drop=True)

    # ✅ Token-Schwierigkeitstabelle
    token_rows = []
    for tok, f1_list in token_perf.items():
        token_rows.append({
            "token": tok,
            "mean_f1": np.mean(f1_list),
            "median_f1": np.median(f1_list),
            "seen_in_subsets": len(f1_list),
        })
    token_difficulty_df = pd.DataFrame(token_rows).sort_values("mean_f1", ascending=False)

    # ✅ Analytischer Text
    analysis = []
    analysis.append(f"Getestete Subsets: {len(subsets)}")

    analysis.append("\n**Leicht zu lernende Tokens (höchste mittlere F1):**")
    for _, r in token_difficulty_df.head(5).iterrows():
        analysis.append(f"  {r['token']}: mean_f1={r['mean_f1']:.3f}")

    analysis.append("\n**Schwer zu lernende Tokens (niedrigste mittlere F1):**")
    for _, r in token_difficulty_df.tail(5).iterrows():
        analysis.append(f"  {r['token']}: mean_f1={r['mean_f1']:.3f}")

    print('\n'.join(analysis))
    return {
        "summary_df": summary_df,
        "token_difficulty_df": token_difficulty_df,
        "analysis_text": "\n".join(analysis),
    }


import random
import torch
from torch.utils.data import Dataset

class MLMDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128, mask_token="[MASK]"):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.mask_token = mask_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        masked = entry["masked"]
        label = entry["label"]
        target = label.lower()

        expected_key = f"[{label}_ARTICLE]"
        if expected_key not in masked:
            raise ValueError(
                f"Expected key '{expected_key}' not found in masked text: {masked}"
            )

        text = masked.replace(expected_key, self.mask_token)

        # Tokenize first to find mask index
        enc_full = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids_full = enc_full.input_ids.squeeze(0)

        # There must be exactly **one** mask token
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
        mask_positions = (input_ids_full == mask_token_id).nonzero(as_tuple=False)

        if mask_positions.size(0) != 1:
            raise ValueError(f"Exactly one mask required, found {mask_positions.size(0)}.")

        mask_index = mask_positions.item()
        seq_len = input_ids_full.size(0)

        # Random left and right context sizes
        # We sample how many tokens to keep before and after the mask
        max_left = min(mask_index, self.max_length // 2)
        max_right = min(seq_len - mask_index - 1, self.max_length // 2)

        left_keep = random.randint(0, max_left)
        right_keep = random.randint(0, max_right)

        start = mask_index - left_keep
        end = mask_index + right_keep + 1  # +1 to include mask token

        # Crop the token sequence
        input_ids_cropped = input_ids_full[start:end]

        # Re-pad + optionally truncate to max_length
        enc = self.tokenizer.pad(
            {"input_ids": input_ids_cropped.unsqueeze(0)},
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc.input_ids.squeeze(0)
        attn_mask = enc.attention_mask.squeeze(0)

        # Target label → single token
        label_ids = self.tokenizer(ENCODING_PREFIX + target, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

        return input_ids, attn_mask, label_ids


def train(
        base_model: str = "dbmdz/german-gpt2",
        # training parameters
        batch_size: int = 4,
        epochs: int = 5,
        lr: float = 1e-4,
        combinations = ('NN', 'NM', 'NF', 'GN', 'GM', 'GF', 'DN', 'DM', 'DF', 'AN', 'AM', 'AF'),
        pooling_length: int = 3,
):

    # ======= Config =======
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = base_model.split("/")[-1]
    output_path = f'results/decoder-mlm-head-gender-de/{"-".join(sorted(combinations))}/{pooling_length}/{model_id}'

    # ======= Prepare Tokenizer =======
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": f"[MASK]"})

    tokenizer.pad_token = tokenizer.eos_token


    # ======= Example Training Data =======
    split = 'train'

    train_datasets = []
    for label in combinations:
        dataset = read_article_ds(article=label, split=split)
        train_datasets.append(dataset)

    labels = set.union(*[set(d['label'].tolist()) for d in train_datasets])
    tokens = [f'{ENCODING_PREFIX}{l.lower()}' for l in labels]
    train_dataset = pd.concat(train_datasets, ignore_index=True)
    train_dataset = train_dataset[train_dataset['token_count'] == 1].reset_index(drop=True)

    if True:
        # balance the datasets by 'dataset_label'
        key = 'dataset_label'
        key = 'label'
        min_size = train_dataset.groupby(key).size().min()
        train_dataset = (
            train_dataset
            .groupby(key, group_keys=False)
            .apply(lambda x: x.sample(min_size, random_state=42))
            .reset_index(drop=True)
        )

    target_token_ids = []
    # ensure each token is a single token
    for token in tokens:
        token_ids = tokenizer(token, add_special_tokens=False)['input_ids']
        if len(token_ids) > 1:
            raise ValueError(f"Token '{token}' is not a single token (token IDs: {token_ids})")
        target_token_ids.extend(token_ids)

    print("Target token IDs:", target_token_ids)

    train_dataset = MLMDataset(tokenizer, train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ======= Model =======

    model = DecoderModelWithMLMHead.from_pretrained(
        base_model,
        mask_token_id=tokenizer.mask_token_id,
        target_token_ids=target_token_ids,
        pooling_length=pooling_length,
    )
    model.decoder.resize_token_embeddings(len(tokenizer))  # In case we added [MASK]

    # Freeze decoder
    for p in model.decoder.parameters():
        p.requires_grad = False

    model.to(DEVICE)

    # ======= Optimizer =======
    model.train()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)

    loss_weights = None


    # ======= Training Loop =======
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids, attn_mask, labels = [b.to(DEVICE) for b in batch]

            # labels: we only care about the mask position
            output = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels, loss_weights=loss_weights)

            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(loss)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f}")

        # ======= Validation Evaluation =======
        metrics = evaluate_split(model, tokenizer, split="val", batch_size=batch_size, combinations=combinations)
        print(f"Validation - Acc: {metrics['acc']:.4f} | Prec: {metrics['prec']:.4f} | "
              f"Rec: {metrics['rec']:.4f} | F1: {metrics['f1']:.4f}")
        print("Macro Precision:", metrics["prec"])
        print(pd.DataFrame(metrics["per_token"]).T)  # tabellarische Übersicht

    # ======= Save Model =======
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    metrics = evaluate_split(model, tokenizer, split="test", batch_size=batch_size, combinations=combinations)
    print(f"Test - Acc: {metrics['acc']:.4f} | Prec: {metrics['prec']:.4f} | "
          f"Rec: {metrics['rec']:.4f} | F1: {metrics['f1']:.4f}")
    print("Macro Precision:", metrics["prec"])
    print(pd.DataFrame(metrics["per_token"]).T)  # tabellarische Übersicht
    # save metrics as csv
    metrics_df = pd.DataFrame.from_dict(metrics["per_token"]).T
    metrics_df.to_csv(os.path.join(output_path, "test_metrics_per_token.csv"))

    score = metrics['f1']
    return score, output_path

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

@torch.no_grad()
def evaluate_split(model, tokenizer, split="val", batch_size=16, combinations=('NN', 'NM', 'NF', 'GN', 'GM', 'GF', 'DN', 'DM', 'DF', 'AN', 'AM', 'AF')):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    datasets = [read_article_ds(article=c, split=split) for c in combinations]
    data = pd.concat(datasets, ignore_index=True)
    data = data[data['token_count'] == 1].reset_index(drop=True)
    dataset = MLMDataset(tokenizer, data)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds, all_labels = [], []
    label_map = {tid: idx for idx, tid in enumerate(model.target_token_ids)}

    for input_ids, attn_mask, labels in loader:
        input_ids, attn_mask, labels = [b.to(DEVICE) for b in (input_ids, attn_mask, labels)]
        output = model(input_ids=input_ids, attention_mask=attn_mask)

        preds = torch.argmax(output.logits, dim=-1).cpu().numpy()
        pred_token_ids = [model.target_token_ids[p] for p in preds]
        pred_tokens = tokenizer.convert_ids_to_tokens(pred_token_ids)

        true_ids = labels[:, 0].cpu().numpy() if labels.ndim > 1 else labels.cpu().numpy()
        true_tokens = tokenizer.convert_ids_to_tokens(true_ids)

        all_preds.extend(pred_tokens)
        all_labels.extend(true_tokens)

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    token_metrics = {
        token: {
            "precision": report[token]["precision"],
            "recall": report[token]["recall"],
            "f1": report[token]["f1-score"],
            "support": report[token]["support"],
        }
        for token in set(y_true)  # oder: model.target_token_ids -> convert_ids_to_tokens
    }

    # Macro
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "per_token": token_metrics,
    }



import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_model_pooling_results(base_dir, model_dirs, show=True, save_path=None):
    init_matplotlib(use_tex=True)

    results = {label: {} for label in model_dirs.keys()}


    # --- Scan numeric pooling folders ---
    pooling_folders = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            pl = int(name)
        except:
            continue
        pooling_folders.append((pl, name))

    pooling_folders.sort()

    # --- Read macro F1 for each model and pooling length ---
    for pl, folder_name in pooling_folders:
        folder_path = os.path.join(base_dir, folder_name)
        for label, model_subfolder in model_dirs.items():
            model_path = os.path.join(folder_path, model_subfolder)
            csv_path = os.path.join(model_path, "test_metrics_per_token.csv")

            if not os.path.exists(csv_path):
                print(f"Missing file for {label} pooling {pl}: {csv_path}")
                results[label][pl] = float("nan")
                continue

            df = pd.read_csv(csv_path, index_col=0)
            macro_f1 = df["f1"].mean()
            results[label][pl] = float(macro_f1)

    # --- Plot ---
    plt.figure(figsize=(4, 2))

    for label, pl_map in results.items():
        sorted_items = sorted(pl_map.items(), key=lambda x: x[0])
        xs = [k for k, _ in sorted_items]
        ys = [v for _, v in sorted_items]

        # normal line
        plt.plot(xs, ys, marker="o", label=label, linewidth=2)

        # --- Highlight maximum point ---
        import numpy as np

        ys_arr = np.array(ys, dtype=float)
        if not np.all(np.isnan(ys_arr)):
            max_idx = np.nanargmax(ys_arr)
            max_x = xs[max_idx]
            max_y = ys_arr[max_idx]

            plt.scatter(
                [max_x],
                [max_y],
                s=90,
                marker="o",
                edgecolor="black",
                linewidth=1.2,
                zorder=5,
            )

    plt.xlabel("Pooling Length")
    plt.ylabel("Macro F1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return results


if __name__ == "__main__":
    all_combinations = ('NN', 'NM', 'NF', 'GN', 'GM', 'GF', 'DN', 'DM', 'DF', 'AN', 'AM', 'AF')
    combination_id = "-".join(sorted(all_combinations))
    base = f"results/decoder-mlm-head-gender-de/{combination_id}"

    models = [
        'meta-llama/Llama-3.2-3B',
        'dbmdz/german-gpt2',
    ]

    for model in models:
        scores = {}
        for pooling_length in range(1, 8):
            score, output = train(base_model=model, combinations=all_combinations, pooling_length=pooling_length)
            scores[pooling_length] = score

        print("Pooling Length Scores:", scores)
        print("Best Pooling Length:", max(scores, key=scores.get), "with score", scores[max(scores, key=scores.get)])
        # save best model with pooling length in folder model
        best_pl = max(scores, key=scores.get)
        output_folder = output.split('/')[:-2] + [model]
        output_folder = '/'.join(output_folder)
        best_model_folder = f"{base}/{best_pl}/{model.split('/')[-1]}"
        os.makedirs(output_folder, exist_ok=True)
        if os.path.exists(best_model_folder):
            import shutil
            shutil.copytree(best_model_folder, output_folder, dirs_exist_ok=True)
            print(f"Saved best model to {output_folder}")


        results = plot_model_pooling_results(
            base_dir=base,
            model_dirs={
                r"\gpttwo": "german-gpt2",
                r"\llama": "Llama-3.2-3B"
            },
            save_path='img/de_pooling_length_comparison.pdf'
        )
