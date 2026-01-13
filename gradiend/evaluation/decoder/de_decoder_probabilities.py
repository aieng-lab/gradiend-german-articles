import os
import time

from gradiend.evaluation.xai.io import all_interesting_classes, GradiendGenderCaseConfiguration, Case, Gender, \
    statistical_analysis_config_classes, article_mapping

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from scipy.spatial.distance import jensenshannon

from gradiend.evaluation.decoder.de_decoder_analysis import DeDecoderAnalysis
from gradiend.model import ModelWithGradiend
from gradiend.training.decoder_only_mlm.model import DecoderModelWithMLMHead


# ============================================================
#  Probability delta helper (core scientific metric)
# ============================================================

def mean_delta(df_before, df_after, token):
    """Mean probability difference for a token column."""
    col = f"prob_{token}" if f"prob_{token}" in df_after.columns else token
    return float((df_after[col] - df_before[col]).mean())

def mean_delta_subset(base_df, mod_df, token, subset):
    """Mean delta for a token restricted to rows whose ds is in the subset."""
    col = f"prob_{token}" if f"prob_{token}" in base_df.columns else token
    dfb = base_df[base_df["ds"].isin(subset)]
    dfm = mod_df[mod_df["ds"].isin(subset)]
    return float((dfm[col].to_numpy() - dfb[col].to_numpy()).mean())


# ============================================================
#  LM collapse: Jensen–Shannon distance
# ============================================================

def js_collapse(neutral_before, neutral_after):
    """
    Measures LM drift on a neutral datasets.
    Lower = better stability.
    """
    cols = [c for c in neutral_before.columns if c.startswith("prob_")]
    P = neutral_before[cols].to_numpy()
    Q = neutral_after[cols].to_numpy()

    js_vals = [jensenshannon(P[i], Q[i], base=2) for i in range(len(P))]
    return float(np.mean(js_vals))


# ============================================================
#  Core LR objective
# ============================================================

def compute_objective(base_eval, mod_eval,
                      base_neutral, mod_neutral,
                      target_token):
    """
    Scientific metric:
        selectivity = Δp_eval - Δp_neutral
    Collapse penalty:
        1 / (1 + JS(neutral_before, neutral_after))
    """
    Δ_eval = mean_delta(base_eval, mod_eval, target_token)
    Δ_neutral = mean_delta(base_neutral, mod_neutral, target_token)

    selectivity = Δ_eval - Δ_neutral

    collapse = js_collapse(base_neutral, mod_neutral)
    collapse_penalty = 1.0 / (1.0 + collapse)

    return float(selectivity * collapse_penalty), selectivity, collapse
# ============================================================
#  Fine-grained deltas per ds
# ============================================================

def per_ds_delta(base_df, mod_df, token):
    """
    Returns a dict:
        { ds_label : mean(mod_df - base_df) for that ds }
    """
    col = f"prob_{token}" if f"prob_{token}" in base_df.columns else token

    out = {}
    for ds in sorted(base_df["ds"].unique()):
        b = base_df[base_df["ds"] == ds][col].to_numpy()
        m = mod_df[mod_df["ds"] == ds][col].to_numpy()
        if len(b) > 0:
            out[ds] = float((m - b).mean())
    return out


# ============================================================
#  Extended evaluation metric (fine-grained + TG/RG separation)
# ============================================================

def compute_objective_extended(
        base_eval, mod_eval,
        base_neutral, mod_neutral,
        target_token,
        TG, RG
):
    # overall deltas
    Δ_eval = mean_delta(base_eval, mod_eval, target_token)
    Δ_neutral = mean_delta(base_neutral, mod_neutral, target_token)

    # TG / RG deltas (coarse)
    Δ_TG = mean_delta_subset(base_eval, mod_eval, target_token, TG)
    Δ_RG = mean_delta_subset(base_eval, mod_eval, target_token, RG)

    # fine-grained per-ds deltas
    Δ_eval_ds = per_ds_delta(base_eval, mod_eval, target_token)

    # selectivities
    Sel_TG = Δ_TG - Δ_neutral
    Sel_RG = Δ_RG - Δ_neutral
    Sel_gap = Sel_TG - Sel_RG

    # LM collapse
    collapse = js_collapse(base_neutral, mod_neutral)

    return {
        "delta_eval": float(Δ_eval),
        "delta_neutral": float(Δ_neutral),

        "delta_TG": float(Δ_TG),
        "delta_RG": float(Δ_RG),

        "Sel_TG": float(Sel_TG),
        "Sel_RG": float(Sel_RG),
        "Sel_gap": float(Sel_gap),

        "collapse": float(collapse),
        "delta_eval_ds": Δ_eval_ds,
    }


# ============================================================
#  NEW LR SELECTION RULE:
#     "take the largest |alpha| where neutral_acc >= threshold"
# ============================================================

def pick_alpha_stability_based(search_records, base_neutral_acc, threshold_ratio=0.95):
    """
    search_records: list of dicts, each containing:
        {"alpha": value, "neutral_acc": acc}
    We pick the alpha with largest |alpha| that satisfies:
        neutral_acc >= threshold_ratio * base_neutral_acc
    """

    allowed = []
    cutoff = base_neutral_acc * threshold_ratio

    for rec in search_records:
        if rec["neutral_acc"] is None:
            continue
        if rec["neutral_acc"] >= cutoff:
            allowed.append(rec)

    if not allowed:
        # fallback: safest small step
        return 0.0

    # choose the one with maximal |alpha|
    best = max(allowed, key=lambda r: abs(r["alpha"]))
    return best["alpha"]


# ============================================================
#  MAIN PIPELINE INCLUDING NEW LR SELECTION
# ============================================================

def evaluate_model_with_ae(
    model_path: str,
    neutral_csv: str,
    output_csv: str,
    feature_factors=None,
    lrs=None,
    target_token='die',
    config=None,
    datasets=None,
):
    if feature_factors is None:
        #feature_factors = [-1, +1]
        # determine suitable feature factor
        metrics = config.get_model_metrics()
        mean_by_class = metrics['mean_by_class']
        config_datasets = config.datasets
        config_dataset_articles = {ds: article_mapping[ds] for ds in config_datasets}
        target_dataset = [ds for ds, art in config_dataset_articles.items() if art == target_token][0]
        other_article_dataset = [ds for ds, art in config_dataset_articles.items() if art != target_token][0]

        target_mean = mean_by_class[target_dataset]
        other_mean = mean_by_class[other_article_dataset]

        if target_mean < other_mean:
            feature_factors = [+1]
        else:
            feature_factors = [-1]

# todo
#    if lrs is None:
    lrs = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 50, 100]
    #lrs = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5] #, 1.0, ] #2.0, 5.0, 10.0, 20, 50, 100]

    base_model = ModelWithGradiend.from_pretrained(model_path, device_decoder=torch.device("cpu"))
    if isinstance(base_model, DecoderModelWithMLMHead):
        base_model = base_model.decoder

    neutral_data = pd.read_csv(neutral_csv)

    base_eval_output_file = f'results/decoder/{base_model.base_model.name_or_path.split("/")[-1]}_base_eval.csv'
    base_neutral_output_file = f'results/decoder/{base_model.base_model.name_or_path.split("/")[-1]}_base_neutral.csv'
    if False or not os.path.exists(base_eval_output_file) or not os.path.exists(base_neutral_output_file):
        base_ana = DeDecoderAnalysis(base_model.base_model, base_model.tokenizer)
        base_eval = base_ana.evaluate_article_probabilities()
        base_neutral = base_ana.evaluate_on_neutral(neutral_data)
        os.makedirs(os.path.dirname(base_eval_output_file), exist_ok=True)
        base_eval.to_csv(base_eval_output_file, index=False)
        base_neutral.to_csv(base_neutral_output_file, index=False)
    else:
        base_eval = pd.read_csv(base_eval_output_file)
        base_neutral = pd.read_csv(base_neutral_output_file)

    # baseline neutral accuracy
    if "label" in base_neutral.columns and "pred" in base_neutral.columns:
        base_neutral_acc = (base_neutral["label"] == base_neutral["pred"]).mean()
    else:
        base_neutral_acc = None


    results = []
    search_log = []
    all_prob_outputs = []
    mod_evals = []
    mod_neutrals = []

    # group definitions available externally in your main()
    # TG = config_datasets
    # RG = datasets - config_datasets

    for g_f in tqdm(feature_factors, desc="feature_factors"):

        per_alpha_records = []

        def evaluate_alpha(alpha):
            enhanced = base_model.modify_model(
                lr=alpha,
                feature_factor=g_f,
                part="decoder",
                top_k=None
            )
            ana = DeDecoderAnalysis(enhanced, base_model.tokenizer)

            start = time.time()
            mod_eval = ana.evaluate_article_probabilities()
            print("Eval time probabilities:", time.time() - start)

            start = time.time()
            mod_neutral = ana.evaluate_on_neutral(neutral_data)
            print("Neutral time probabilities:", time.time() - start)

            start = time.time()
            metrics = compute_objective_extended(
                base_eval,
                mod_eval,
                base_neutral,
                mod_neutral,
                target_token,
                TG=config_datasets,
                RG=list(set(datasets) - set(config_datasets))
            )
            print("Metrics computation time:", time.time() - start)

            # neutral accuracy
            if ("label" in mod_neutral.columns) and ("pred" in mod_neutral.columns):
                neutral_acc = (mod_neutral["label"] == mod_neutral["pred"]).mean()
            else:
                neutral_acc = None

            rec = {
                "alpha": alpha,
                "neutral_acc": neutral_acc,
                "mod_eval": mod_eval,
                "mod_neutral": mod_neutral,
                **metrics,
            }

            mod_neutral["gf"] = g_f
            mod_neutral["alpha"] = alpha
            mod_neutrals.append(mod_neutral)
            mod_eval["gf"] = g_f
            mod_eval["alpha"] = alpha
            mod_evals.append(mod_eval)

            return rec

        # evaluate all alphas
        for alpha in tqdm(lrs, desc=f"g_f={g_f} alphas", leave=False):
            rec = evaluate_alpha(alpha)
            per_alpha_records.append(rec)
            search_log.append({"gf": g_f, **rec})

        # ----------------------------------------------
        # NEW OPTIMIZATION: stability-based selection
        # ----------------------------------------------

        # pick actual record for that alpha
        #best_rec = max(per_alpha_records, key=lambda r: -abs(r["alpha"])
        #               if r["alpha"] == best_alpha else -1e9)

        # save eval probs
        #mod_eval = best_rec["mod_eval"]
        #mod_eval["gf"] = g_f
        #mod_eval["alpha"] = best_alpha
        #all_prob_outputs.append(mod_eval)


    # remove dataframes before saving
    search_log = [{k: v for k, v in rec.items() if k not in ["mod_eval", "mod_neutral"]} for rec in search_log]
    pd.DataFrame(search_log).to_csv(output_csv.replace(".csv", "_search.csv"), index=False)
    #pd.concat(all_prob_outputs).to_csv(output_csv.replace(".csv", "_probs.csv"), index=False)
    pd.concat(mod_evals).to_csv(output_csv.replace(".csv", "_mod_eval.csv"), index=False)
    pd.concat(mod_neutrals).to_csv(output_csv.replace(".csv", "_mod_neutral.csv"), index=False)




if __name__ == "__main__":


    default_evaluation_feature_factors = [-1, +1]
    #default_evaluation_feature_factors = [1]
    #default_evaluation_lrs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    default_evaluation_lrs = [0.0, -0.01, -0.02, -0.05, -0.1, -0.2, -0.5, -1.0, -2.0, -5.0, -10.0]
    #default_evaluation_lrs = [-1.0, ]

    all_interesting_classes_ = {
        ('das', 'die'): [
            (Case.NOMINATIVE, Gender.NEUTRAL, Gender.FEMALE),
            (Case.ACCUSATIVE, Gender.NEUTRAL, Gender.FEMALE),
        ],
    }

    all_interesting_classes_ = {
        ('der', 'die'): [
            (Case.NOMINATIVE, Gender.MALE, Gender.FEMALE),
            (Case.NOMINATIVE, Case.DATIVE, Gender.FEMALE),
            (Case.NOMINATIVE, Case.GENITIVE, Gender.FEMALE),
        ],
    }

    base_model_id = 'bert-base-german-cased'
    base_model_id = 'EuroBERT-210m'
    base_model_id = 'german-gpt2'

    models = [
        #'EuroBERT-210m',
        #'bert-base-german-cased',
        #'german-gpt2',
        #'Llama-3.2-3B',
        #'gbert-large',
        'ModernGBERT_1B',
    ]

    statistical_analysis_config_classes = {
        #('der', 'die'): [
        #    (Case.NOMINATIVE, Gender.MALE, Gender.FEMALE),
            # (Case.NOMINATIVE, Case.DATIVE, Gender.FEMALE),
            # (Case.NOMINATIVE, Case.GENITIVE, Gender.FEMALE),
        #    (Case.DATIVE, Case.ACCUSATIVE, Gender.FEMALE),
        #    (Case.GENITIVE, Case.ACCUSATIVE, Gender.FEMALE),
        #],
        ('der', 'dem'): [
        #    (Case.NOMINATIVE, Case.DATIVE, Gender.MALE),
            # (Case.DATIVE, Gender.MALE, Gender.FEMALE),
            (Case.DATIVE, Gender.NEUTRAL, Gender.FEMALE),
        ],
        ('der', 'des'): [
            (Case.NOMINATIVE, Case.GENITIVE, Gender.MALE),
            # (Case.GENITIVE, Gender.FEMALE, Gender.MALE),
            (Case.GENITIVE, Gender.NEUTRAL, Gender.FEMALE),
        ]
    }

    transitions = {
        ('der', 'die'): 'die',
        ('der', 'dem'): 'dem',
        ('der', 'des'): 'des',
    }

    for base_model_id in models:
        for articles, config_keys in statistical_analysis_config_classes.items():
            print("==== Evaluating model:", base_model_id, "articles:", articles)
            configs = [GradiendGenderCaseConfiguration(*config_key, model_id=base_model_id) for config_key in config_keys]
            datasets = list(set.union(*[set(config.datasets) for config in configs]))

            for config in configs:
                try:
                    model_path = config.gradiend_dir
                    config_datasets = config.datasets

                    neutral_csv = 'data/der_die_das/neutral/neutral_dwk.csv'

                    output = f'{model_path}_decoder_sig_analysis.csv'
                    output2 = f'{model_path}_decoder_sig_analysis_mod_eval.csv'
                    if False or not os.path.exists(output) or not os.path.exists(output2):
                        evaluate_model_with_ae(
                            model_path=model_path,
                            output_csv=output,
                            #feature_factors=default_evaluation_feature_factors,
                            lrs=default_evaluation_lrs,
                            neutral_csv=neutral_csv,
                            config=config,
                            target_token=transitions[articles],
                            datasets=datasets,
                        )
                    else:
                        print(f"Skipping evaluation for {model_path}, output already exists.")
                except NotImplementedError as e:
                    print(e)

        print("==== Finished model:", base_model_id)