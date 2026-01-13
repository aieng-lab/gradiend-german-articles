import seaborn as sns
from gradiend.evaluation.xai.io import Case, Gender, GradiendGenderCaseConfiguration, article_mapping, \
    all_interesting_classes, statistical_analysis_config_classes, pretty_model_mapping
from gradiend.model import GradiendModel, ModelWithGradiend
from gradiend.util import RESULTS_DIR

import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera
from scipy.stats import ttest_ind
import numpy as np
from scipy.stats import ttest_rel



def permutation_test_mean_diff(x, y, n_perm=10000, random_state=0):
    """
    Permutation test for difference in means (independent samples).
    """
    rng = np.random.default_rng(random_state)

    x = np.asarray(x)
    y = np.asarray(y)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    obs = x.mean() - y.mean()

    pooled = np.concatenate([x, y])
    n_x = len(x)

    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        perm_diffs[i] = pooled[:n_x].mean() - pooled[n_x:].mean()

    p = np.mean(np.abs(perm_diffs) >= abs(obs))
    return obs, p


def normality_scores_paired(base, mod):
    """
    Compute normality diagnostics for paired t-test deltas.

    Returns a dict with interpretable scores.
    """
    base = np.asarray(base)
    mod  = np.asarray(mod)
    n = min(len(base), len(mod))
    delta = mod[:n] - base[:n]

    delta = delta[~np.isnan(delta)]

    return {
        "n": len(delta),
        "mean": float(delta.mean()),
        "std": float(delta.std(ddof=1)),
        "skewness": float(skew(delta)),
        "excess_kurtosis": float(kurtosis(delta, fisher=True)),
        "jarque_bera_p": float(jarque_bera(delta)[1]),
    }


# ------------------------------------------------------------
#  Generic helpers
# ------------------------------------------------------------

def _get_token_col(df, token):
    """Return the name of the column holding the prob for `token`."""
    col_prob = f"prob_{token}"
    if col_prob in df.columns:
        return col_prob
    if token in df.columns:
        return token
    raise ValueError(f"Token column for '{token}' not found in df.")


def cohen_d_from_delta(delta):
    """Cohen's d based on delta = mod - base."""
    delta = np.asarray(delta)
    delta = delta[~np.isnan(delta)]
    if delta.size < 2:
        return np.nan
    return float(delta.mean() / (delta.std(ddof=1) + 1e-12))


def paired_effect(base, mod, n_perm=10000, random_state=0):
    """
    Paired stats: mod - base.
    Returns mean_delta, t, p (t-test), p_perm (permutation), d, n.
    """
    rng = np.random.default_rng(random_state)

    base = np.asarray(base)
    mod = np.asarray(mod)

    n = min(len(base), len(mod))
    base = base[:n]
    mod = mod[:n]

    delta = mod - base
    delta = delta[~np.isnan(delta)]

    mean_delta = float(delta.mean())

    # classical paired t-test (kept for comparability)
    t_stat, p_val = ttest_rel(mod[:len(delta)], base[:len(delta)], nan_policy="omit")

    # permutation test on the mean (sign-flip test)
    signs = rng.choice([-1, 1], size=(n_perm, len(delta)))
    perm_means = (signs * delta).mean(axis=1)
    count = np.sum(np.abs(perm_means) >= abs(mean_delta))
    p_perm = float((count + 1) / (n_perm + 1))

    d = cohen_d_from_delta(delta)

    return {
        "n": int(len(delta)),
        "mean_delta": mean_delta,
        "t": float(t_stat),
        "p": float(p_val),          # original t-test p
        "p_perm": p_perm,           # robust alternative
        "cohen_d": d,
    }



def group_delta(base_df, mod_df, token, ds_subset=None):
    """
    Extract paired deltas for a token, optionally restricted to ds_subset.
    """
    col = _get_token_col(base_df, token)

    if ds_subset is not None:
        mask_base = base_df["ds"].isin(ds_subset)
        mask_mod = mod_df["ds"].isin(ds_subset)
        base_vals = base_df.loc[mask_base, col].to_numpy()
        mod_vals = mod_df.loc[mask_mod, col].to_numpy()
    else:
        base_vals = base_df[col].to_numpy()
        mod_vals = mod_df[col].to_numpy()

    n = min(len(base_vals), len(mod_vals))
    return mod_vals[:n] - base_vals[:n]


def neutral_delta(base_neutral_df, mod_neutral_df, token):
    """
    Deltas for neutral datasets (no ds column).
    """
    col = _get_token_col(base_neutral_df, token)
    base_vals = base_neutral_df[col].to_numpy()
    mod_vals = mod_neutral_df[col].to_numpy()
    n = min(len(base_vals), len(mod_vals))
    return mod_vals[:n] - base_vals[:n]



def analyze_noun_level_effects(base_eval, mod_eval, target_token, min_n=5, datasets=None):
    """
    Compute per-noun deltas, significance, effect size.

    Args:
        base_eval: dataframe with columns ['noun', 'ds', ..., prob_token]
        mod_eval: same as base_eval but after modification
        target_token: e.g. "die" or "der"
        min_n: minimum number of occurrences required for stable stats

    Returns:
        noun_stats_df: DataFrame with rows per noun
    """

    if datasets:
        base_eval = base_eval[base_eval["ds"].isin(datasets)].reset_index(drop=True)
        mod_eval = mod_eval[mod_eval["ds"].isin(datasets)].reset_index(drop=True)

    col = f"prob_{target_token}" if f"prob_{target_token}" in base_eval.columns else target_token

    # merge on same rows — they should align by index after evaluation
    merged = pd.DataFrame({
        "noun": base_eval["noun"],
        "base": base_eval[col],
        "mod": mod_eval[col],
        "ds": base_eval["ds"],
    })

    noun_rows = []

    for noun, df in merged.groupby("noun"):
        if len(df) < min_n:
            continue

        b = df["base"].to_numpy()
        m = df["mod"].to_numpy()
        delta = m - b

        # paired test
        t, p = ttest_rel(m, b, nan_policy="omit")
        mean_delta = float(delta.mean())
        std = delta.std(ddof=1)
        d = float(mean_delta / (std + 1e-12)) if std > 0 else np.nan

        noun_rows.append({
            "noun": noun,
            "n": len(df),
            "mean_delta": mean_delta,
            "t": float(t),
            "p": float(p),
            "cohen_d": d,
        })

    noun_stats_df = pd.DataFrame(noun_rows)
    if noun_stats_df.empty:
        return noun_stats_df
    noun_stats_df = noun_stats_df.sort_values("mean_delta", ascending=False).reset_index(drop=True)
    return noun_stats_df

def categorize_nouns(noun_stats_df, alpha=0.05, min_effect=0.002):
    """
    Assign categories based on statistical significance and effect direction/magnitude.
    """

    rows = []
    for _, row in noun_stats_df.iterrows():
        if row["p"] < alpha:
            if row["mean_delta"] > min_effect:
                cat = "strong_positive_effect"
            elif row["mean_delta"] < -min_effect:
                cat = "strong_negative_effect"
            else:
                cat = "weak_effect_but_significant"
        else:
            if abs(row["mean_delta"]) < min_effect:
                cat = "stable_unaffected"
            else:
                cat = "nonsignificant_shift"

        new_row = row.to_dict()
        new_row["category"] = cat
        rows.append(new_row)

    return pd.DataFrame(rows)




def determine_best_gf_alpha(
    mod_eval: pd.DataFrame,
    mod_neutral: pd.DataFrame,
    TG,
    target_token: str,
    accuracy_threshold: float = 0.99,
    base_eval=None,
    alpha_sig_level: float = 0.01,     # unused here
    alpha_p_key: str = "p_perm",       # unused here
    gf: float = None,
):
    if gf is None:
        gfs = mod_neutral["gf"].dropna().unique()
        if len(gfs) != 1:
            raise ValueError(
                f"Expected exactly one gf in mod_neutral when gf is None, got {len(gfs)}: {gfs.tolist()}"
            )
        gf = gfs[0]

    # ------------- LM drop threshold on neutral (for this gf) -------------
    neutral_gf = mod_neutral[mod_neutral["gf"] == gf].copy()
    if neutral_gf.empty:
        raise ValueError(f"No neutral rows found for gf={gf}")

    def accuracy_score(sub_df):
        return (sub_df["label"] == sub_df["pred"]).mean()

    def perplexity_score(sub_df):
        return sub_df["perplexity"].mean()

    is_clm = "perplexity" in neutral_gf.columns
    lms_score = perplexity_score if is_clm else accuracy_score

    # NEW: threshold differs by model type
    threshold_factor = 0.90 if is_clm else float(accuracy_threshold)

    if "alpha" not in neutral_gf.columns:
        raise ValueError("mod_neutral must contain an 'alpha' column.")

    neutral_gf["abs_alpha"] = neutral_gf["alpha"].abs()

    base_df = neutral_gf[neutral_gf["abs_alpha"] == 0]
    if base_df.empty:
        raise ValueError("Neutral baseline (alpha==0) missing; cannot determine LM drop.")
    base_score = float(lms_score(base_df))

    # score per |alpha| (aggregating both signs if present)
    try:
        score_per_abs = neutral_gf.groupby("abs_alpha").apply(lms_score, include_groups=False)
    except TypeError:
        score_per_abs = neutral_gf.groupby("abs_alpha").apply(lms_score)
    score_per_abs = score_per_abs.sort_index()

    def violates(score: float) -> bool:
        if not np.isfinite(score):
            return True
        if is_clm:
            # lower perplexity is better; violation means perplexity got too high
            return float(score) > float(base_score) / float(threshold_factor)
        else:
            # higher accuracy is better; violation means accuracy too low
            return float(score) < float(threshold_factor) * float(base_score)

    # Determine last valid |alpha| before first violation
    abs_grid = score_per_abs.index.to_numpy(dtype=float)
    nonzero_abs = abs_grid[abs_grid > 0]

    drop_abs = None
    if len(nonzero_abs) > 0:
        first_abs = float(nonzero_abs[0])
        if not violates(float(score_per_abs.loc[first_abs])):
            last_valid = first_abs
            for i in range(len(nonzero_abs) - 1):
                nxt = float(nonzero_abs[i + 1])
                if violates(float(score_per_abs.loc[nxt])):
                    break
                last_valid = nxt
            drop_abs = last_valid

    # valid alphas = all non-zero alphas with |alpha| <= drop_abs
    all_alphas = np.array(sorted(set(neutral_gf["alpha"].unique().tolist())), dtype=float)
    nonzero_alphas = all_alphas[all_alphas != 0.0]
    if len(nonzero_alphas) == 0:
        raise ValueError("No non-zero alphas available for this gf.")

    # fallback if every non-zero violates
    if drop_abs is None:
        alpha = float(nonzero_alphas[np.argmin(np.abs(nonzero_alphas))])
        print(f"All non-zero alphas violate LM threshold; using smallest non-zero alpha: {alpha}")
        return gf, alpha

    valid_alphas = [float(a) for a in nonzero_alphas if abs(float(a)) <= float(drop_abs)]
    if not valid_alphas:
        alpha = float(nonzero_alphas[np.argmin(np.abs(nonzero_alphas))])
        print(f"No valid alphas before LM drop; using smallest non-zero alpha: {alpha}")
        return gf, alpha

    # ------------- choose alpha by max mean target prob on TG -------------
    eval_tg = mod_eval[(mod_eval["ds"].isin(TG)) & (mod_eval["gf"] == gf)].copy()
    if eval_tg.empty:
        alpha = float(nonzero_alphas[np.argmin(np.abs(nonzero_alphas))])
        print(f"No TG rows found for gf={gf}; using smallest non-zero alpha: {alpha}")
        return gf, alpha

    prob_col = f"prob_{target_token}" if f"prob_{target_token}" in eval_tg.columns else target_token
    if prob_col not in eval_tg.columns:
        raise ValueError(f"Target probability column not found: tried '{prob_col}'")

    best_alpha, best_mean = None, -np.inf
    for a in valid_alphas:
        sub = eval_tg[eval_tg["alpha"] == a]
        if sub.empty:
            continue
        m = float(sub[prob_col].mean())
        # tie-break: prefer smaller |alpha|
        if (m > best_mean) or (np.isclose(m, best_mean) and best_alpha is not None and abs(a) < abs(best_alpha)):
            best_mean = m
            best_alpha = float(a)

    if best_alpha is None:
        alpha = float(nonzero_alphas[np.argmin(np.abs(nonzero_alphas))])
        print(f"No TG data for valid alphas; using smallest non-zero alpha: {alpha}")
        return gf, alpha

    print(
        f"Alpha selected (valid: |alpha|<= {drop_abs:g} before LM drop; "
        f"max mean P({target_token}) on TG): {best_alpha} (mean={best_mean:.6g})"
    )
    return gf, float(best_alpha)



def change_model(
        model_path: str,
        out_model_path: str,
        mod_eval_csv: str,
        mod_neutral_csv: str,
        target_token: str,
        TG,
        accuracy_threshold: float = 0.99,
):
    """
    Change Gradiend model to have specified alpha and gf.

    Args:
        model_path: path to original model
        out_model_path: path to save modified model
    """

    mod_eval  = pd.read_csv(mod_eval_csv)
    mod_neutral  = pd.read_csv(mod_neutral_csv)

    gf, alpha = determine_best_gf_alpha(
        mod_eval,
        mod_neutral,
        TG,
        target_token,
        accuracy_threshold=accuracy_threshold,
    )

    model_with_gradiend = ModelWithGradiend.from_pretrained(model_path)
    tokenizer = model_with_gradiend.tokenizer
    modified_model = model_with_gradiend.modify_model(lr=alpha, feature_factor=gf)

    modified_model.save_pretrained(out_model_path, safe_serialization=True)
    tokenizer.save_pretrained(out_model_path)
    print(f"Modified model saved to {out_model_path} with alpha={alpha}, gf={gf}.")


def analyze_steering_from_files(
    base_eval_csv: str,
    mod_eval_csv: str,
    base_neutral_csv: str,
    mod_neutral_csv: str,
    target_token: str,
    TG,
    RG,
    other,
    out_prefix: str = None,
    out_modified_model: str = None,
    acc_threshold: float = 0.99,
):
    """
    Post-hoc analysis for one model/config.

    Args:
        base_eval_csv: path to baseline decoder_article_probs (before steering)
        mod_eval_csv:  path to *_mod_eval.csv  (after steering; best alpha)
        base_neutral_csv: baseline neutral probs
        mod_neutral_csv:  steered neutral probs
        target_token: e.g. "die" or "der"
        TG: iterable of ds labels (trained datasets)
        RG: iterable of ds labels (related, not trained)
        out_prefix: if not None, CSVs will be written with this prefix

    Returns:
        group_stats_df, ds_stats_df, compare_stats_df
    """

    # ------------- load data -------------
    base_eval = pd.read_csv(base_eval_csv)
    mod_eval  = pd.read_csv(mod_eval_csv)
    base_neutral = pd.read_csv(base_neutral_csv)
    mod_neutral  = pd.read_csv(mod_neutral_csv)

    gf, alpha = determine_best_gf_alpha(
        mod_eval,
        mod_neutral,
        TG,
        target_token,
        accuracy_threshold=acc_threshold,
        base_eval=base_eval
    )

    # --------- NEW: alpha-sweep robustness + selection sensitivity (writes CSV + plots) ----------
    if False and out_prefix is not None:
        sweep_df, selection_df, selected = run_alpha_robustness_bundle(
            base_eval=base_eval,
            mod_eval=mod_eval,              # full (unfiltered)
            base_neutral=base_neutral,
            mod_neutral=mod_neutral,        # full (unfiltered)
            TG=TG,
            gf=gf,
            target_token=target_token,
            accuracy_threshold=acc_threshold,
            alpha_sig_level=0.001,
            alpha_p_key="p_perm",
            out_prefix=out_prefix,
            title_prefix=f"{target_token} (gf={gf})",
        )
        print("Alpha selection sensitivity:")
        print(selection_df)


    #
    #base_eval = base_eval[(base_eval['alpha'] == alpha) & (base_eval['gf'] == gf)].reset_index(drop=True)
    mod_eval  = mod_eval[(mod_eval['alpha'] == alpha) & (mod_eval['gf'] == gf)].reset_index(drop=True)
    #base_neutral = base_neutral[(base_neutral['alpha'] == alpha) & (base_neutral['gf'] == gf)].reset_index(drop=True)
    mod_neutral  = mod_neutral[(mod_neutral['alpha'] == alpha) & (mod_neutral['gf'] == gf)].reset_index(drop=True)

    if mod_eval.empty:
        raise ValueError(f"No mod_eval data found for alpha={alpha}, gf={gf}.")

    if mod_neutral.empty:
        raise ValueError(f"No mod_neutral data found for alpha={alpha}, gf={gf}.")

    # sanity
    if "ds" not in base_eval.columns:
        raise ValueError("base_eval must contain a 'ds' column.")
    if "ds" not in mod_eval.columns:
        raise ValueError("mod_eval must contain a 'ds' column.")

    TG = list(TG)
    RG = list(RG)

    # ------------- group-level stats -------------
    group_rows = []

    # ALL eval
    delta_all = group_delta(base_eval, mod_eval, target_token, ds_subset=None)


    base_eval_all_others = base_eval[base_eval["ds"].isin(other)].reset_index(drop=True)
    mod_eval_all_others = mod_eval[mod_eval["ds"].isin(other)].reset_index(drop=True)
    stats_all = paired_effect(
        base_eval_all_others[_get_token_col(base_eval_all_others, target_token)],
        mod_eval_all_others[_get_token_col(mod_eval_all_others, target_token)],
    )
    stats_all.update({"group": "OA"})
    group_rows.append(stats_all)

    # TG
    for tg in TG:
        delta_TG = group_delta(base_eval, mod_eval, target_token, ds_subset=[tg])
        base_TG = base_eval[base_eval["ds"] == tg][_get_token_col(base_eval, target_token)]
        mod_TG  = mod_eval[mod_eval["ds"] == tg][_get_token_col(mod_eval, target_token)]
        stats_TG = paired_effect(base_TG, mod_TG)
        stats_TG.update({"group": "Train: " + tg})
        group_rows.append(stats_TG)

    # RG
    for rg in RG:
        delta_RG = group_delta(base_eval, mod_eval, target_token, ds_subset=[rg])
        base_RG = base_eval[base_eval["ds"] == rg][_get_token_col(base_eval, target_token)]
        mod_RG  = mod_eval[mod_eval["ds"] == rg][_get_token_col(mod_eval, target_token)]
        stats_RG = paired_effect(base_RG, mod_RG)
        stats_RG.update({"group": rg})
        group_rows.append(stats_RG)

    # neutral
    delta_neutral = neutral_delta(base_neutral, mod_neutral, target_token)
    stats_neutral = paired_effect(
        base_neutral[_get_token_col(base_neutral, target_token)],
        mod_neutral[_get_token_col(mod_neutral, target_token)],
    )
    stats_neutral.update({"group": "Neutral"})
    group_rows.append(stats_neutral)


    #violinplot_standardized_deltas(
    #    base_list=[base_TG, base_RG, base_neutral['prob_masked']],
    #    mod_list=[mod_TG, mod_RG, mod_neutral['prob_masked']],
    #    labels=["TG", "RG", "Neutral"]
    #)

    #scores_TG = normality_scores_paired(base_TG, mod_TG)
    #scores_RG = normality_scores_paired(base_RG, mod_RG)
    #scores_N = normality_scores_paired(base_neutral['prob_masked'], mod_neutral['prob_masked'])

    p_key = 'p_perm'

    group_stats_df = pd.DataFrame(group_rows)

    # map group stats to output formats
    def significance_mapper_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    # returns "< 10^-n" for small p-values for largest integer n
#    def significance_mapper_tens(p):

    # map p to <10-n

    def significance_mapper(p):
        stars = significance_mapper_stars(p)

        if p < 0.001:
            # Auto scientific notation (e.g., 1.2e-31)
            p_str = f"{p:.1e}".replace("e-0", "e-").replace("e-", "\\times 10^{-").replace("}", "}")
            # Optionally: use simpler LaTeX formatting:
            p_str = f"{p:.1e}".replace("e", "\\times 10^{") + "}"
            return f"{stars} (${p_str}$)"
        else:
            return f"{stars} (${p:.3f}$)"

    group_stats_output = group_stats_df.copy()
    group_stats_output["significance"] = group_stats_output[p_key].map(significance_mapper)
    columns = ['group', 'n', 'mean_delta', 'cohen_d', 'significance']
    group_stats_output = group_stats_output[columns]

    # ------------- per-datasets stats -------------
    ds_rows = []
    col = _get_token_col(base_eval, target_token)

    for ds in sorted(base_eval["ds"].unique()):
        base_ds = base_eval[base_eval["ds"] == ds][col]
        mod_ds  = mod_eval[mod_eval["ds"] == ds][col]
        stats = paired_effect(base_ds, mod_ds)
        stats.update({"ds": ds})
        ds_rows.append(stats)

    ds_stats_df = pd.DataFrame(ds_rows)

    # ------------- group comparisons (difference-of-deltas) -------------
    compare_rows = []

    if (delta_TG.size > 1) and (delta_RG.size > 1):
        # classical Welch t-test (keep)
        t_TR, p_TR = ttest_ind(
            delta_TG, delta_RG, equal_var=False, nan_policy="omit"
        )

        diff_mean = float(delta_TG.mean() - delta_RG.mean())

        pooled = np.sqrt(
            (delta_TG.var(ddof=1) + delta_RG.var(ddof=1)) / 2.0 + 1e-12
        )
        d_TR = float(diff_mean / pooled)

        # permutation test (robust)
        _, p_perm_TR = permutation_test_mean_diff(delta_TG, delta_RG)

        compare_rows.append({
            "comparison": "TG_vs_RG",
            "mean_diff": diff_mean,
            "t": float(t_TR),
            "p": float(p_TR),  # Welch t-test
            "p_perm": p_perm_TR,  # robust alternative
            "cohen_d": d_TR,
            "n1": int(delta_TG.size),
            "n2": int(delta_RG.size),
        })

    # TG vs NEUTRAL (delta_TG vs delta_neutral)
    if (delta_TG.size > 1) and (delta_neutral.size > 1):
        t_TN, p_TN = ttest_ind(delta_TG, delta_neutral, equal_var=False, nan_policy="omit")
        diff_mean = float(delta_TG.mean() - delta_neutral.mean())
        pooled = np.sqrt(
            (delta_TG.var(ddof=1) + delta_neutral.var(ddof=1)) / 2.0 + 1e-12
        )
        d_TN = float(diff_mean / pooled)
        compare_rows.append({
            "comparison": "TG_vs_NEUTRAL",
            "mean_diff": diff_mean,
            "t": float(t_TN),
            "p": float(p_TN),
            "cohen_d": d_TN,
            "n1": int(delta_TG.size),
            "n2": int(delta_neutral.size),
        })

    # RG vs NEUTRAL
    if (delta_RG.size > 1) and (delta_neutral.size > 1):
        t_RN, p_RN = ttest_ind(delta_RG, delta_neutral, equal_var=False, nan_policy="omit")
        diff_mean = float(delta_RG.mean() - delta_neutral.mean())
        pooled = np.sqrt(
            (delta_RG.var(ddof=1) + delta_neutral.var(ddof=1)) / 2.0 + 1e-12
        )
        d_RN = float(diff_mean / pooled)
        compare_rows.append({
            "comparison": "RG_vs_NEUTRAL",
            "mean_diff": diff_mean,
            "t": float(t_RN),
            "p": float(p_RN),
            "cohen_d": d_RN,
            "n1": int(delta_RG.size),
            "n2": int(delta_neutral.size),
        })

    compare_stats_df = pd.DataFrame(compare_rows)

    noun_stats = analyze_noun_level_effects(base_eval, mod_eval, target_token=target_token, datasets=RG)
    noun_stats = categorize_nouns(noun_stats)


    # ------------- optional saving -------------
    if out_prefix is not None:
        group_stats_df.to_csv(out_prefix + "_group_stats.csv", index=False)
        ds_stats_df.to_csv(out_prefix + "_ds_stats.csv", index=False)
        compare_stats_df.to_csv(out_prefix + "_group_comparisons.csv", index=False)
        noun_stats.to_csv(out_prefix + "_noun_level_stats.csv", index=False)

    return group_stats_df, group_stats_output, ds_stats_df, compare_stats_df, noun_stats


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def robust_zscore(x):
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / mad

def violinplot_standardized_deltas(base_list, mod_list, labels, clip=10):
    """
    Violinplots of robust-standardized paired deltas.
    Focuses on distribution shape (normality check), not scale.
    """
    deltas_z = []

    for base, mod in zip(base_list, mod_list):
        base = np.asarray(base)
        mod = np.asarray(mod)
        n = min(len(base), len(mod))
        delta = mod[:n] - base[:n]
        z = robust_zscore(delta)
        z = np.clip(z, -clip, clip)   # purely visual
        deltas_z.append(z)

    plt.figure(figsize=(6, 4))
    sns.violinplot(deltas_z)
    plt.axhline(0, linestyle="--", color="black", linewidth=1)

    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("robust z-score of Δ")
    plt.title("Shape of paired-delta distributions")
    plt.tight_layout()
    plt.show()



def violinplot_paired_deltas(base_list, mod_list, labels):
    """
    Violinplot of paired deltas (mod - base), exactly as used for ttest_rel.

    Args:
        base_list: list of 1D arrays (baseline values)
        mod_list:  list of 1D arrays (modified values)
        labels:    list of group names
    """
    assert len(base_list) == len(mod_list) == len(labels)

    deltas = []
    for base, mod in zip(base_list, mod_list):
        base = np.asarray(base)
        mod = np.asarray(mod)
        n = min(len(base), len(mod))
        deltas.append(mod[:n] - base[:n])

    plt.figure(figsize=(6, 4))
    sns.violinplot(deltas)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Δ (mod − base)")
    plt.title("Paired deltas used in t-tests")
    plt.tight_layout()
    plt.show()


import os
import matplotlib.pyplot as plt

def _neutral_score_and_valid(neutral_gf: pd.DataFrame, accuracy_threshold: float):
    """
    Returns:
      is_clm (bool), lms_score_fn, base_neutral_lms (float),
      acc_per_alpha (pd.Series indexed by alpha),
      valid_alphas (pd.Series filtered, alpha!=0)
    """
    def accuracy_score(sub_df):
        return (sub_df["label"] == sub_df["pred"]).mean()

    def perplexity_score(sub_df):
        return sub_df["perplexity"].mean()

    is_clm = "perplexity" in neutral_gf.columns
    lms_score = perplexity_score if is_clm else accuracy_score
    base_neutral_lms = lms_score(neutral_gf[neutral_gf["alpha"] == 0])

    try:
        acc_per_alpha = neutral_gf.groupby("alpha").apply(lms_score, include_groups=False)
    except TypeError:
        acc_per_alpha = neutral_gf.groupby("alpha").apply(lms_score)

    if is_clm:
        valid_alphas = acc_per_alpha[acc_per_alpha <= base_neutral_lms / accuracy_threshold]
    else:
        valid_alphas = acc_per_alpha[acc_per_alpha >= accuracy_threshold * base_neutral_lms]

    valid_alphas = valid_alphas[valid_alphas.index != 0]
    return is_clm, lms_score, float(base_neutral_lms), acc_per_alpha, valid_alphas


def _paired_base_mod_series_fast(df_base: pd.DataFrame, df_mod: pd.DataFrame, value_col: str):
    """
    Fast-ish pairing:
      - prefer merging on stable key cols (object/int/bool) excluding common float-ish columns.
      - fallback to positional alignment.
    """
    drop_cols = {value_col, "alpha", "gf", "label", "pred", "perplexity"}
    common_cols = [c for c in df_base.columns if c in df_mod.columns and c not in drop_cols]

    key_cols = []
    for c in common_cols:
        dt0, dt1 = df_base[c].dtype, df_mod[c].dtype
        if dt0 != dt1:
            continue
        if (pd.api.types.is_object_dtype(dt0)
            or pd.api.types.is_integer_dtype(dt0)
            or pd.api.types.is_bool_dtype(dt0)):
            key_cols.append(c)

    if key_cols:
        s0 = df_base.set_index(key_cols)[value_col]
        s1 = df_mod.set_index(key_cols)[value_col]
        s0a, s1a = s0.align(s1, join="inner")
        if len(s0a) > 0:
            return s0a, s1a

    # positional fallback
    b = df_base[value_col].reset_index(drop=True)
    m = df_mod[value_col].reset_index(drop=True)
    n = min(len(b), len(m))
    return b.iloc[:n], m.iloc[:n]


def compute_alpha_sweep_stats(
    base_eval: pd.DataFrame,
    mod_eval: pd.DataFrame,
    base_neutral: pd.DataFrame,
    mod_neutral: pd.DataFrame,
    TG,
    gf,
    target_token: str,
    accuracy_threshold: float = 0.99,
    alpha_sig_level: float = 0.001,
    alpha_p_key: str = "p_perm",
):
    """
    Computes per-alpha stats for:
      - TG effect (mean_delta, p_perm, cohen_d)
      - Neutral effect + neutral score constraint validity
    Returns a DataFrame with one row per alpha.
    """
    TG = list(TG)

    # restrict to gf
    me = mod_eval[mod_eval["gf"] == gf].copy()
    mn = mod_neutral[mod_neutral["gf"] == gf].copy()

    # token columns
    col_eval_base = _get_token_col(base_eval, target_token)
    col_eval_mod  = _get_token_col(me, target_token)
    col_neu_base  = _get_token_col(base_neutral, target_token)
    col_neu_mod   = _get_token_col(mn, target_token)

    # neutral validity / score
    is_clm, _, base_neutral_lms, acc_per_alpha, valid_alphas = _neutral_score_and_valid(
        neutral_gf=mn, accuracy_threshold=accuracy_threshold
    )

    # alpha universe (include 0 for reference)
    alphas = sorted([float(a) for a in me["alpha"].unique()])

    rows = []
    for a in alphas:
        # --- Neutral score and validity ---
        neu_score = float(acc_per_alpha.loc[a]) if a in acc_per_alpha.index else np.nan
        is_valid = bool(a in valid_alphas.index) if a != 0 else True

        # --- TG stats vs base_eval ---
        base_tg_df = base_eval[base_eval["ds"].isin(TG)]
        mod_tg_df  = me[(me["alpha"] == a) & (me["ds"].isin(TG))]

        if base_tg_df.empty or mod_tg_df.empty:
            tg_stats = {"n": 0, "mean_delta": np.nan, "p_perm": np.nan, "cohen_d": np.nan}
        else:
            s0, s1 = _paired_base_mod_series_fast(base_tg_df, mod_tg_df, col_eval_base)
            # s1 column name might differ; use values
            tg_stats = paired_effect(np.asarray(s0), np.asarray(s1),)
        tg_p = tg_stats.get(alpha_p_key, tg_stats.get("p_perm", np.nan))
        tg_sig = bool(np.isfinite(tg_p) and float(tg_p) < float(alpha_sig_level))

        # --- Neutral stats vs base_neutral ---
        base_n_df = base_neutral
        mod_n_df  = mn[mn["alpha"] == a]
        if base_n_df.empty or mod_n_df.empty:
            n_stats = {"n": 0, "mean_delta": np.nan, "p_perm": np.nan, "cohen_d": np.nan}
        else:
            s0n, s1n = _paired_base_mod_series_fast(base_n_df, mod_n_df, col_neu_base)
            n_stats = paired_effect(np.asarray(s0n), np.asarray(s1n))
        n_p = n_stats.get(alpha_p_key, n_stats.get("p_perm", np.nan))
        n_sig = bool(np.isfinite(n_p) and float(n_p) < float(alpha_sig_level))

        rows.append({
            "alpha": float(a),
            "neutral_score": neu_score,             # accuracy or perplexity depending on model type
            "neutral_base_score": float(base_neutral_lms),
            "neutral_is_clm": bool(is_clm),
            "neutral_valid": bool(is_valid),

            "tg_n": int(tg_stats.get("n", 0)),
            "tg_mean_delta": float(tg_stats.get("mean_delta", np.nan)),
            "tg_p_perm": float(tg_p) if tg_p is not None else np.nan,
            "tg_sig": bool(tg_sig),
            "tg_cohen_d": float(tg_stats.get("cohen_d", np.nan)),

            "neutral_n": int(n_stats.get("n", 0)),
            "neutral_mean_delta": float(n_stats.get("mean_delta", np.nan)),
            "neutral_p_perm": float(n_p) if n_p is not None else np.nan,
            "neutral_sig": bool(n_sig),
            "neutral_cohen_d": float(n_stats.get("cohen_d", np.nan)),
        })

    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def select_alpha_sensitivity_rules(
    sweep_df: pd.DataFrame,
    alpha_sig_level: float = 0.01,
):
    """
    Returns dict rule_name -> alpha (float).
    Rules:
      A: min |alpha| among neutral_valid & TG significant
      B: median |alpha| among neutral_valid & TG significant
      C: max |alpha| among neutral_valid (upper end of safe regime)
    Fallbacks:
      - If no TG-significant valid alphas: A/B fall back to C.
    """
    df = sweep_df.copy()
    df = df[df["alpha"] != 0].reset_index(drop=True)

    valid = df[df["neutral_valid"]].copy()
    tg_sig_valid = valid[valid["tg_sig"]].copy()

    def _max_abs_alpha(d):
        if d.empty:
            return None
        idx = (d["alpha"].abs()).idxmax()
        return float(d.loc[idx, "alpha"])

    def _min_abs_alpha(d):
        if d.empty:
            return None
        d2 = d.copy()
        d2["abs_alpha"] = d2["alpha"].abs()
        d2 = d2.sort_values(["abs_alpha", "alpha"]).reset_index(drop=True)
        return float(d2.loc[0, "alpha"])

    def _median_abs_alpha(d):
        if d.empty:
            return None
        d2 = d.copy()
        d2["abs_alpha"] = d2["alpha"].abs()
        d2 = d2.sort_values(["abs_alpha", "alpha"]).reset_index(drop=True)
        return float(d2.loc[len(d2)//2, "alpha"])

    alpha_C = _max_abs_alpha(valid) or float(df["alpha"].iloc[df["alpha"].abs().idxmax()])

    alpha_A = _min_abs_alpha(tg_sig_valid) if not tg_sig_valid.empty else alpha_C
    alpha_B = _median_abs_alpha(tg_sig_valid) if not tg_sig_valid.empty else alpha_C

    return {
        "A_min_abs_TGsig_valid": float(alpha_A),
        "B_median_abs_TGsig_valid": float(alpha_B),
        "C_max_abs_valid": float(alpha_C),
    }


def plot_alpha_sweep(
    sweep_df: pd.DataFrame,
    selected_alphas: dict,
    out_path_prefix: str,
    title_prefix: str = "",
):
    """
    Writes two plots:
      1) TG mean_delta vs |alpha| (significance/validity markers)
      2) Neutral score vs |alpha| with validity markers
    """
    df = sweep_df.copy()
    df = df.sort_values("alpha")
    df["abs_alpha"] = df["alpha"].abs()

    # --- Plot 1: TG effect ---
    plt.figure(figsize=(7.2, 4.2))
    x = df["abs_alpha"].to_numpy()
    y = df["tg_mean_delta"].to_numpy()
    plt.plot(x, y, marker="o", linewidth=1.5)

    # mark valid/invalid points
    for _, r in df.iterrows():
        if r["alpha"] == 0:
            continue
        if not r["neutral_valid"]:
            plt.scatter([abs(r["alpha"])], [r["tg_mean_delta"]], marker="x", s=60)
        elif r["tg_sig"]:
            plt.scatter([abs(r["alpha"])], [r["tg_mean_delta"]], marker="s", s=45)

    # vertical lines for rules
    for name, a in selected_alphas.items():
        plt.axvline(abs(a), linestyle="--", linewidth=1.0)
        plt.text(abs(a), plt.gca().get_ylim()[1], f" {name}", rotation=90, va="top", fontsize=8)

    plt.axhline(0, linestyle="--", linewidth=1.0)
    plt.xlabel("|alpha|")
    plt.ylabel("TG mean Δp (mod - base)")
    plt.title((title_prefix + " " if title_prefix else "") + "Alpha sweep: TG effect")
    plt.tight_layout()
    plt.xscale("log")
    plt.savefig(out_path_prefix + "_alpha_sweep_tg.png", dpi=200)
    plt.savefig(out_path_prefix + "_alpha_sweep_tg.pdf")
    plt.show()
    plt.close()

    # --- Plot 2: Neutral score ---
    plt.figure(figsize=(7.2, 4.2))
    y2 = df["neutral_score"].to_numpy()
    plt.plot(x, y2, marker="o", linewidth=1.5)

    # mark invalid points
    for _, r in df.iterrows():
        if r["alpha"] == 0:
            continue
        if not r["neutral_valid"]:
            plt.scatter([abs(r["alpha"])], [r["neutral_score"]], marker="x", s=60)

    # reference line at base neutral score
    base_score = float(df["neutral_base_score"].iloc[0]) if len(df) > 0 else np.nan
    if np.isfinite(base_score):
        plt.axhline(base_score, linestyle="--", linewidth=1.0)

    for name, a in selected_alphas.items():
        plt.axvline(abs(a), linestyle="--", linewidth=1.0)

    plt.xlabel("|alpha|")
    plt.ylabel("Neutral score (acc↑ or ppl↓)")
    plt.title((title_prefix + " " if title_prefix else "") + "Alpha sweep: Neutral performance")
    plt.tight_layout()
    # log x scale
    plt.xscale("log")
    plt.savefig(out_path_prefix + "_alpha_sweep_neutral.png", dpi=200)
    plt.savefig(out_path_prefix + "_alpha_sweep_neutral.pdf")
    plt.show()
    plt.close()


def run_alpha_robustness_bundle(
    base_eval: pd.DataFrame,
    mod_eval: pd.DataFrame,
    base_neutral: pd.DataFrame,
    mod_neutral: pd.DataFrame,
    TG,
    gf,
    target_token: str,
    accuracy_threshold: float,
    alpha_sig_level: float = 0.01,
    alpha_p_key: str = "p_perm",
    out_prefix: str = None,
    title_prefix: str = "",
):
    """
    One-call convenience wrapper:
      - compute sweep df
      - compute 3 rule alphas
      - write CSV + plots (if out_prefix provided)
    Returns (sweep_df, selection_df, selected_alphas_dict)
    """
    sweep_df = compute_alpha_sweep_stats(
        base_eval=base_eval,
        mod_eval=mod_eval,
        base_neutral=base_neutral,
        mod_neutral=mod_neutral,
        TG=TG,
        gf=gf,
        target_token=target_token,
        accuracy_threshold=accuracy_threshold,
        alpha_sig_level=alpha_sig_level,
        alpha_p_key=alpha_p_key,
    )

    selected = select_alpha_sensitivity_rules(sweep_df, alpha_sig_level=alpha_sig_level)
    selection_df = pd.DataFrame([{"rule": k, "alpha": v} for k, v in selected.items()])

    if out_prefix is not None:
        sweep_df.to_csv(out_prefix + "_alpha_sweep.csv", index=False)
        selection_df.to_csv(out_prefix + "_alpha_selection_sensitivity.csv", index=False)
        plot_alpha_sweep(
            sweep_df=sweep_df,
            selected_alphas=selected,
            out_path_prefix=out_prefix,
            title_prefix=title_prefix,
        )

    return sweep_df, selection_df, selected



import os
import pandas as pd

# ---- CONFIG YOU SHOULD ADAPT ONCE ----

# Order of dataset columns in the LaTeX header:
#   key = how it appears in group_stats_output["group"]
#   title = LaTeX macro shown in the header
TABLE_GROUPS = [
    ("dataNM",   r"\dataNM"),
    ("dataGF",   r"\dataGF"),
    ("dataDF",   r"\dataDF"),
    ("Neutral",  r"\dataNEUT"),   # group label is literally "Neutral" in your code
]

# Map model_id to the macro you want in the table
MODEL_LATEX = pretty_model_mapping

def latex_escape(s: str) -> str:
    # minimal escaping (underscores are the usual culprit)
    return s.replace("_", r"\_")

def fmt_num(x, ndigits=4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{float(x):.{ndigits}f}"

def get_group_triplet(group_stats_output: pd.DataFrame, ds_name: str, TG_list) -> tuple[str, str, str]:
    """
    Returns (mean_delta, cohen_d, significance) for a dataset group.
    Handles TG being labeled as "Train: <ds>" in group_stats_output.
    """
    candidates = []
    if ds_name == "Neutral":
        candidates = ["Neutral"]
    else:
        ds_name = ds_name.removeprefix('data')
        # If ds is in TG, analyze_steering_from_files labels it "Train: <ds>"
        if ds_name in TG_list:
            candidates = [f"Train: {ds_name}", ds_name]
        else:
            candidates = [ds_name, f"Train: {ds_name}"]

    row = None
    for c in candidates:
        hit = group_stats_output[group_stats_output["group"] == c]
        if not hit.empty:
            row = hit.iloc[0]
            break

    if row is None:
        return ("--", "--", "--")

    return (fmt_num(row.get("mean_delta")), fmt_num(row.get("cohen_d")), row.get("significance", "--") or "--")

def config_to_latex_name(config) -> str:
    # Prefer an explicit LaTeX macro if you have one
    for attr in ("latex_macro", "latex", "latex_name"):
        if hasattr(config, attr) and getattr(config, attr):
            return getattr(config, attr)
    # Fallback: use id
    if hasattr(config, "id"):
        return latex_escape(str(config.id))
    return latex_escape(str(config))

def make_table_header() -> list[str]:
    # colspec: Model, ArtTrans, alpha, then 4*(ΔP d Sig) = 12 cols
    lines = []
    lines.append(r"\begin{tabular}{llr" + "rrr"*len(TABLE_GROUPS) + r"}\toprule")
    # top multi-columns
    group_titles = " & ".join([rf"\multicolumn{{3}}{{c}}{{\textbf{{{title}}}}}" for _, title in TABLE_GROUPS])
    lines.append(r"& & & " + group_titles + r" \\")
    # cmidrules
    # columns: 1 Model, 2 Art, 3 alpha, then groups start at 4
    start = 4
    cm = []
    for _ in TABLE_GROUPS:
        cm.append(rf"\cmidrule(lr){{{start}-{start+2}}}")
        start += 3
    lines.append(" ".join(cm))
    # second header row
    sub = []
    for _ in TABLE_GROUPS:
        sub += [r"$\Delta \mathbb{P}$", r"$d$", r"\textbf{Sig.}"]
    lines.append(r"\textbf{Model} & \textbf{Art. Trans.} & $\alpha$ & " + " & ".join(sub) + r" \\ \midrule")
    return lines

def generate_latex_table_for_model(base_model_id: str,
                                   statistical_analysis_config_classes,
                                   article_mapping,
                                   RESULTS_DIR,
                                   analyze_steering_from_files,
                                   determine_best_gf_alpha,
                                   acc_threshold: float = 0.99) -> str:
    """
    Generates ONE LaTeX tabular for a given base_model_id, including:
      - base row
      - one row per config in statistical_analysis_config_classes
    """
    model_cell = MODEL_LATEX.get(base_model_id, latex_escape(base_model_id))
    lines = make_table_header()

    # Base row (empty stats)
    empty_triplets = ["--"] * (3 * len(TABLE_GROUPS))
    lines.append(f"{model_cell} & -- & 0.0 & " + " & ".join(empty_triplets) + r" \\")

    # All configs
    for articles, config_keys in statistical_analysis_config_classes.items():
        configs = [GradiendGenderCaseConfiguration(*config_key, model_id=base_model_id) for config_key in config_keys]

        for config in configs:
            try:
                model_path = config.gradiend_dir
                config_datasets = config.datasets
            except FileNotFoundError:
                continue

            for article in config.articles:
                if article != 'der':
                    continue

                other_article = [a for a in config.articles if a != article][0]

                this_dataset = [ds for ds in config_datasets if article_mapping[ds] == article]
                other_configs_datasets_with_same_article = [
                    ds for ds in article_mapping.keys()
                    if article_mapping[ds] == article and ds not in this_dataset
                ]
                other_article_datasets = [ds for ds, art in article_mapping.items() if art not in config.articles]

                # --- compute alpha (and gf) the same way analyze_steering_from_files does ---
                base_eval = pd.read_csv(f"results/decoder/{base_model_id}_base_eval.csv")
                mod_eval  = pd.read_csv(f"{model_path}_decoder_sig_analysis_mod_eval.csv")
                base_neut = pd.read_csv(f"results/decoder/{base_model_id}_base_neutral.csv")
                mod_neut  = pd.read_csv(f"{model_path}_decoder_sig_analysis_mod_neutral.csv")

                gf, alpha = determine_best_gf_alpha(
                    mod_eval, mod_neut, this_dataset, other_article,
                    accuracy_threshold=acc_threshold,
                    base_eval=base_eval
                )

                # --- run analysis to get group_stats_output (already contains significance strings) ---
                group_stats, group_stats_output, *_ = analyze_steering_from_files(
                    base_eval_csv=f"results/decoder/{base_model_id}_base_eval.csv",
                    mod_eval_csv=f"{model_path}_decoder_sig_analysis_mod_eval.csv",
                    base_neutral_csv=f"results/decoder/{base_model_id}_base_neutral.csv",
                    mod_neutral_csv=f"{model_path}_decoder_sig_analysis_mod_neutral.csv",
                    target_token=other_article,
                    TG=this_dataset,
                    RG=other_configs_datasets_with_same_article,
                    other=other_article_datasets,
                    out_prefix=None,
                    acc_threshold=acc_threshold,
                )

                # drop OA
                group_stats_output = group_stats_output[group_stats_output["group"] != "OA"].reset_index(drop=True)

                # Build stats cells for the fixed table groups
                triplets = []
                for ds_key, _title in TABLE_GROUPS:
                    md, d, sig = get_group_triplet(group_stats_output, ds_key, TG_list=list(this_dataset))

                    md = float(md) * 100
                    md = f"{md:.2f}"

                    d = float(d)
                    d = f"{d:.2f}"

                    sig = sig.split('(')[0].strip()
                    if not sig:
                        sig = 'n.s.'

                    if ds_key.removeprefix('data') in this_dataset:
                        # make bold if in TG
                        md = rf"\textbf{{{md}}}"
                        d  = rf"\textbf{{{d}}}"
                        sig = rf"\textbf{{{sig}}}"

                    triplets.extend([md, d, sig])

                # Row formatting like your example: " , + <configmacro>"
                cfg_name = config.pretty_model_id
                model_col = r"\, + " + cfg_name

                art_trans = rf"${article}\!\to\!{other_article}$"
                lines.append(
                    f"{model_col} & {art_trans} & {alpha} & " + " & ".join(triplets) + r" \\"
                )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    models = [
        'bert-base-german-cased',
        'gbert-large',
        'EuroBERT-210m',
        'german-gpt2',
        'Llama-3.2-3B',
        'ModernGBERT_1B',
    ]

    for base_model_id in models:
        acc_threshold = 0.99 if 'bert' in base_model_id.lower() else 0.99
        print(generate_latex_table_for_model(
             base_model_id=base_model_id,
             statistical_analysis_config_classes=statistical_analysis_config_classes,
             article_mapping=article_mapping,
             RESULTS_DIR=RESULTS_DIR,
             analyze_steering_from_files=analyze_steering_from_files,
             determine_best_gf_alpha=determine_best_gf_alpha,
             acc_threshold=acc_threshold
        ))

        for articles, config_keys in statistical_analysis_config_classes.items():
            configs = [GradiendGenderCaseConfiguration(*config_key, model_id=base_model_id) for config_key in config_keys]
            datasets = list(set.union(*[set(config.datasets) for config in configs]))

            for config in configs:

                try:
                    model_path = config.gradiend_dir
                    config_datasets = config.datasets
                except FileNotFoundError:
                    print(f"Model path for config {config.id} not found, skipping.")
                    continue

                for article in config.articles:
                    other_article = [a for a in config.articles if a != article][0]
                    this_dataset = [ds for ds in config_datasets if article_mapping[ds] == article]
                    other_configs_datasets_with_same_article = [ds for ds in article_mapping.keys() if article_mapping[ds] == article and ds not in this_dataset]
                    other_article_datasets = [ds for ds, article in article_mapping.items() if article not in config.articles]

                    modified_model_path = f"{RESULTS_DIR}/changed_models/{config.id}_{base_model_id}_to_{other_article}"

                    if not os.path.exists(modified_model_path):
                        change_model(
                            model_path=model_path,
                            out_model_path=modified_model_path,
                            mod_eval_csv=f"{model_path}_decoder_sig_analysis_mod_eval.csv",
                            mod_neutral_csv=f"{model_path}_decoder_sig_analysis_mod_neutral.csv",
                            target_token=other_article,
                            TG=this_dataset,
                            accuracy_threshold=acc_threshold,
                        )
