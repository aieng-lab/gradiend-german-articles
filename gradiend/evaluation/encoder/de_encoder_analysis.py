import ast
import json
import os

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import torch
from tqdm import tqdm
from gradiend.data.util import get_file_name, json_dumps
from gradiend.evaluation.encoder.encoder_analysis import (
    EncoderAnalysis,
    get_pearson_correlation,
    get_spearman_correlation,
    z_score,
)
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from scipy.stats import f_oneway

from gradiend.training.decoder_only_mlm.model import DecoderModelWithMLMHead
from gradiend.util import get_files_and_folders_with_prefix


class DeEncoderAnalysis(EncoderAnalysis):
    def __init__(self, config):
        self.config = config
        super().__init__(config)

    def analyse_encoder(
        self,
        model_with_gradiend,
        dataset,
        output,
        plot=False,
        multi_task=False,
        shared=False,
        full=False,
        neutral_data=None,
    ):
        from gradiend.combined_models.combined_gradiends import StackedGradiend

        model = model_with_gradiend.base_model
        tokenizer = model_with_gradiend.tokenizer
        mask_token = tokenizer.mask_token

        combined_gradiend = None
        if hasattr(model_with_gradiend, "gradient") and isinstance(
            model_with_gradiend.gradient, StackedGradiend
        ):
            combined_gradiend = model_with_gradiend.gradient

        combinations = dataset['dataset_label'].unique().tolist() if full else None
        cache_default_predictions_dict = self.read_default_predictions(model, combinations=combinations)

        modified_cache = []

        def get_de_default_predictions(masked_text):
            if masked_text in cache_default_predictions_dict:
                return cache_default_predictions_dict[masked_text]

            predictions = self.evaluate_determiners(model, tokenizer, masked_text)
            cache_default_predictions_dict[masked_text] = predictions

            if not modified_cache:
                modified_cache.append(True)
            return predictions

        if combined_gradiend is not None:
            source = combined_gradiend.source
        else:
            source = model_with_gradiend.gradiend.kwargs["training"]["source"]

        filled_texts = []
        correctly_filled_texts = []
        default_preds = self.config["default_predictions"]

        def process_entry(row, plot=False):

            key = row["dataset_label"]
            masked = row["masked"]
            label = row["label"]
            encoded_values = []
            articles = []
            labels = []
            mask_labels = []
            dataset_labels = []
            default_predictions = {k: [] for k in default_preds}

            inputs = []
            masked_texts = []

            template_key = f"[{label}_ARTICLE]"
            filled_text = masked.replace(template_key, mask_token)
            filled_texts.append(filled_text)
            correctly_filled_texts.append(filled_text.replace(mask_token, label.lower()))

            label = row["label"].lower()

            # TODO own function that returns the encoded value
            if source == "diff":
                label_factual = label
                label_counter_factual = self.config[key]["inverse"]

                inputs_factual = model_with_gradiend.create_inputs(
                    filled_text, label_factual
                )
                grads_factual = model_with_gradiend.forward_pass(
                    inputs_factual, return_dict=False
                )
                inputs_counter_factual = model_with_gradiend.create_inputs(
                    filled_text, label_counter_factual
                )
                grads_counter_factual = model_with_gradiend.forward_pass(
                    inputs_counter_factual, return_dict=False
                )
                grads = grads_factual - grads_counter_factual
                inputs.append(grads)
                encoded = model_with_gradiend.gradiend.encoder(grads).item()
            else:
                if source == "gradient":
                    masked_label = label
                elif source == "inv_gradient":
                    masked_label = self.config[key]["inverse"]
                else:
                    raise ValueError(f"Unknown source: {source}")
                mask_labels.append(masked_label)
                inputs.append((filled_text, masked_label))

                # if combined_gradiend:
                #     encoded = combined_gradiend.forward(filled_text, label=masked_label)
                # else:
                if multi_task and source == "inv_gradient":
                    for label in masked_label:
                        encoded = (
                            model_with_gradiend.encode(filled_text, label=label)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        encoded_values.append(encoded)
                        articles.append(row["label"])
                        dataset_labels.append(row["dataset_label"])

                        labels.append([label] * row["token_count"])
                        default_prediction = get_de_default_predictions(filled_text)
                        default_prediction["label"] = label

                        for key, value in default_prediction.items():
                            default_predictions[key].append(value)

                        masked_texts.append(masked)

                else:
                    if shared:
                        encoded = (
                            model_with_gradiend.encode(
                                filled_text, label=masked_label, shared=False
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        encoded = (
                            model_with_gradiend.encode(filled_text, label=masked_label)
                            .detach()
                            .cpu()
                            .tolist()
                        )

                    encoded_values.append(encoded)

                    articles.append(row["label"])
                    dataset_labels.append(row["dataset_label"])

                    labels.append([label] * row["token_count"])

                    default_prediction = get_de_default_predictions(filled_text)

                    default_prediction["label"] = label

                    for key, value in default_prediction.items():
                        if key in default_predictions:
                            default_predictions[key].append(value)

                    # checking for missing keys
                    for key in [k for k in default_predictions if k not in {'label', 'most_likely_token'}]:
                        if key not in default_prediction:
                            raise ValueError(f"Key {key} not in default prediction for text: {filled_text}")

                    masked_texts.append(masked)

                unique_labels = [list(item) for item in set(tuple(x) for x in labels)]

            results = pd.DataFrame(
                {
                    "text": masked_texts,
                    "state": articles,
                    "dataset_labels": dataset_labels,
                    "encoded": encoded_values,
                    "labels": unique_labels,
                    "type": f"{self.det_combination} masked",
                    "mask_labels": mask_labels,
                    **default_predictions,
                }
            )

            results["state_value"] = results["dataset_labels"].map(
                lambda dataset_label: self.config[dataset_label]["code"]
            )
            # if not combined_gradiend:
            #     results['z_score'] = z_score(results['encoded'])
            results = results.sort_values(by="state")

            if plot:
                plt.title(row["masked"])
                sns.boxplot(x="state", y="z_score", data=results)
                plt.show()

            return results

        tqdm.pandas(desc=f"Analyze with {self.det_combination} Test Data")
        dataset[f"{self.det_combination}"] = dataset.progress_apply(
            process_entry, axis=1
        )

        results = dataset[f"{self.det_combination}"].tolist()

        texts = []
        encoded_values = []
        labels = []
        default_predictions = {k: [] for k in default_preds}

        tokens_to_ignore = set(self.config["token_to_ignore"])
        ingore_tokens = list(
            set(
                (
                    token
                    for det in tokens_to_ignore
                    for token in tokenizer(det, add_special_tokens=False)["input_ids"]
                )
            )
        )


        torch.manual_seed(42)
        if neutral_data is None:
            neutral_data = correctly_filled_texts
        # TODO this right now is not that important, i dont have the right datasets for this.

        try:
            if isinstance(model, DecoderModelWithMLMHead):
                # run using CLM gradients of the underlying decoder-only model
                model_with_gradiend.base_model = model_with_gradiend.base_model.decoder

            for text in tqdm(
                neutral_data, desc=f"{self.det_combination} data without determiners masked"
            ):
                encoded, masked_text, label = model_with_gradiend.mask_and_encode(
                    text,
                    ignore_tokens=ingore_tokens,
                    return_masked_text=True,
                    shared=shared,
                )
                texts.append(text)
                encoded_values.append(encoded.tolist())
                labels.append(label)

                default_prediction = get_de_default_predictions(masked_text)
                default_prediction["label"] = label
                for key, value in default_prediction.items():
                    if key in default_predictions:
                        default_predictions[key].append(value)
        except Exception as e:
            print(f"Error processing neutral data: {e}")
            raise e

        result = pd.DataFrame(
            {
                "text": texts,
                "state": None,
                "dataset_labels": None,
                "encoded": encoded_values,
                "type": f"no {self.det_combination} masked",
                **default_predictions,
            }
        )
        results.append(result)

        if modified_cache:
            self.write_default_predictions(cache_default_predictions_dict, model)

        total_results = pd.concat(results)

        # if not combined_gradiend:
        #     mean = total_results['encoded'].mean()
        #     std = total_results['encoded'].std()
        #     total_results['global_z_score'] = (total_results['encoded'] - mean) / std

        for article in self.articles:
            total_results[article] = total_results[article].apply(json_dumps)

        total_results["label"] = total_results["label"].apply(json_dumps)
        total_results["most_likely_token"] = total_results["most_likely_token"].apply(
            json_dumps
        )

        total_results.to_csv(output, index=False)

        if plot:
            # plot results
            self.plot_model_results(total_results, title=output.removesuffix(".csv"))

            plot_results = (
                total_results[total_results["type"] == f"{self.det_combination} masked"]
                .sort_values(by="state")
                .reset_index(drop=True)
            )
            sns.boxplot(x="state", y="encoded", data=plot_results)
            plt.title(model_with_gradiend.name_or_path)
            plt.show()

            cor = np.nanmean(
                [
                    text_df[["encoded", "state_value"]].corr(method="pearson")[
                        "encoded"
                    ]["state_value"]
                    for text, text_df in plot_results.groupby("text")
                ]
            )
            print("Correlation", cor)

            plot_results_MF = plot_results[plot_results["state"] != "B"]
            cor = np.nanmean(
                [
                    text_df[["encoded", "state_value"]].corr(method="pearson")[
                        "encoded"
                    ]["state_value"]
                    for text, text_df in plot_results_MF.groupby("text")
                ]
            )
            print("Correlation MF", cor)

        return total_results

    def read_eval_results(self, *encoded_values, prefix=None, suffix=".csv", **kwargs):
        if prefix:
            # find all models in the folder with the suffix
            encoded_values = list(encoded_values) + get_files_and_folders_with_prefix(
                prefix, suffix=suffix
            )

        if len(encoded_values) > 1:
            metrics = {}
            for ev in encoded_values:
                m = self.get_model_metrics(ev, **kwargs)
                metrics[ev] = m

            return metrics

        raw_encoded_values = encoded_values[0]

        encoded_values = get_file_name(raw_encoded_values, file_format="csv", **kwargs)
        json_file = encoded_values.replace(".csv", ".json")

        try:
            return json.load(open(json_file, "r+"))
        except FileNotFoundError:
            print("Computing model metrics for", encoded_values)

        df_all = pd.read_csv(encoded_values)

        return df_all, json_file

    def get_model_metrics(
        self, *encoded_values, prefix=None, multi_grad=False, suffix=".csv", split=None, **kwargs
    ):
        # TODO also use helper function for this
        if prefix:
            encoded_values = list(encoded_values) + get_files_and_folders_with_prefix(
                prefix, suffix=suffix
            )

        if len(encoded_values) > 1:
            metrics = {}
            for ev in encoded_values:
                m = self.get_model_metrics(ev, **kwargs)
                metrics[ev] = m

            return metrics

        raw_encoded_values = encoded_values[0]

        scores = None
        encoded_values = get_file_name(raw_encoded_values, file_format="csv", split=split, inverse=None, **kwargs)
        json_file = encoded_values.replace(".csv", ".json")

        scores = None
        if os.path.exists(json_file):
            print("Loading existing metrics from", json_file)
            with open(json_file, "r") as f:
                scores = json.load(f)

            if 'pearson' not in scores:
                scores = None
            if scores['pearson'] is None or np.isnan(scores['pearson']):
                scores = None
                print("Scores contain NaN, recomputing", encoded_values)

        if scores is None:
            for use_inverse in [None, True, False]:
                _kwargs = {} if use_inverse is None else {'variant': 'all'}
                encoded_values = get_file_name(raw_encoded_values, file_format="csv", split=split, inverse=use_inverse, **kwargs, **_kwargs)

                df_all = pd.read_csv(
                    encoded_values,
                    converters={
                        "encoded": lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x
                    },
                )

                if use_inverse is None:
                    scores = self._get_basic_model_metrics(df_all=df_all)
                elif use_inverse is True:
                    scores['all_inverse'] = self._get_basic_model_metrics(df_all=df_all)
                else:
                    scores['all_id'] = self._get_basic_model_metrics(df_all=df_all)

            print(scores)

            with open(json_file, "w") as f:
                json.dump(scores, f, indent=4)

        return scores

    def _get_basic_model_metrics(self, df_all, multi_grad=False):
        try:
            df = df_all[df_all["type"] == f"{self.det_combination} masked"]
        except KeyError:
            df = df_all

        df[f"z_score_{self.det_combination}"] = z_score(
            df, key="encoded", groupby="text"
        )
        df[f"global_z_score_{self.det_combination}"] = z_score(df["encoded"])

        # TODO the acc_M_pos/new can be calculated using the dataset_labels, more robust...
        # df['state_value'] = df['state_value'].apply(lambda x: 1 if x in [0,1,2,3] else 0)

        if multi_grad:
            df["state_value"] = df["state_value"].apply(
                lambda x: 1 if x in [0, 2] else 2 if x in [4, 6] else 0
            )
        else:
            df["state_value"] = df["state_value"].apply(
                lambda x: 1 if x in [0, 1, 2, 3] else 0 if x in [4, 5, 6, 7] else 2
            )

        acc_M_positive = np.mean(
            [
                ((text_df["encoded"] >= 0) == text_df["state_value"].astype(bool)).sum()
                / len(text_df)
                for text, text_df in df.groupby("text")
            ]
        )
        acc_M_negative = np.mean(
            [
                ((text_df["encoded"] < 0) == text_df["state_value"].astype(bool)).sum()
                / len(text_df)
                for text, text_df in df.groupby("text")
            ]
        )


        # rename the keys such that blanks are replaced by '_'
        df_all = df_all.rename(columns=lambda x: x.replace(" ", "_"))
        encoded_abs_means = (
            df_all.groupby("type")["encoded"]
            .apply(lambda group: group.abs().mean())
            .to_dict()
        )
        encoded_means = (
            df_all.groupby("type")["encoded"]
            .apply(lambda group: group.mean())
            .to_dict()
        )

        # map encoded values to the predicted class, i.e. >= 0.5 -> female, <= -0.5 -> male, >-0.5 & <0.5 -> neutral

        df_all["predicted_female_pos"] = df_all["encoded"].apply(
            lambda x: 1 if x >= 0.5 else (-1 if x <= -0.5 else 0)
        )
        df_all["predicted_male_pos"] = df_all["encoded"].apply(
            lambda x: 1 if x <= -0.5 else (-1 if x >= 0.5 else 0)
        )

        df_all_labels = (
            df_all["dataset_labels"]
            .apply(lambda x: self.config.get(x, {}).get("encoding", 0))
            .astype(int)
        )
        df_labels = (
            df["dataset_labels"]
            .apply(lambda x: self.config.get(x, {}).get("encoding", 0))
            .astype(int)
        )

        df_all["state_value"] = df_all_labels
        df["state_value"] = df_labels

        balanced_acc_female_pos = balanced_accuracy_score(
            df_all["predicted_female_pos"], df_all_labels
        )
        balanced_acc_male_pos = balanced_accuracy_score(
            df_all["predicted_male_pos"], df_all_labels
        )
        acc_total = max(balanced_acc_female_pos, balanced_acc_male_pos)

        pearson_total = get_pearson_correlation(df_all)
        spearman_total = get_spearman_correlation(df_all)

        pearson = get_pearson_correlation(df)
        spearman = get_spearman_correlation(df)


        mean_by_class = df.groupby("dataset_labels")["encoded"].mean()
        mean_by_label = df.groupby("label")["encoded"].mean()

        scores = {
            "pearson_total": pearson_total["correlation"],
            "pearson_total_p_value": pearson_total["p_value"],
            "spearman_total": spearman_total["correlation"],
            "spearman_total_p_value": spearman_total["p_value"],
            "acc_total": acc_total,
            "pearson": abs(pearson["correlation"]),
            "pearson_p_value": pearson["p_value"],
            "spearmann": spearman,
            "spearman_p_value": spearman["p_value"],
            "acc": max(acc_M_negative, acc_M_positive),
            "encoded_abs_means": encoded_abs_means,
            "encoded_means": encoded_means,
            "mean_by_class": mean_by_class.to_dict(),
            "mean_by_label": mean_by_label.to_dict(),
            **self.get_std_stats(df),
        }

        print(scores)

        return scores

    def get_model_metrics_m_dim(
        self, *encoded_values, prefix=None, multi_grad=False, suffix=".csv", **kwargs
    ):

        df_all, json_file = self.read_eval_results(
            *encoded_values, prefix=prefix, suffix=suffix, **kwargs
        )

        df_aggregated = df_all.copy()
        df_aggregated["encoded"] = df_aggregated["encoded"].apply(ast.literal_eval)
        df_aggregated["encoded"] = df_aggregated["encoded"].apply(lambda x: np.mean(x))

        score_aggregated = self._get_basic_model_metrics(
            df_all=df_aggregated, multi_grad=multi_grad
        )
        results = {}

        try:
            df = df_all[df_all["type"] == f"{self.det_combination} masked"]
        except KeyError:
            df = df_all


        df["state_value"] = df["state_value"].apply(
            lambda x: 1 if x in [0, 1, 2, 3] else 0 if x in [4, 5, 6, 7] else 2
        )
        df["encoded"] = df["encoded"].apply(ast.literal_eval)

        X = np.array(df["encoded"].tolist())
        y = np.array(df["state_value"].tolist())

        pearson_per_dim = {}
        for dim in range(X.shape[1]):
            score = -pearsonr(X[:, dim], y).correlation
            pearson_per_dim[dim] = score
        results["pearson_per_dimension"] = pearson_per_dim

        categories = np.unique(y)
        anova_results = {}
        for dim in range(X.shape[1]):
            groups = [X[y == cat, dim] for cat in categories]
            f_val, p_val = f_oneway(*groups)
            anova_results[f"dim_{dim}"] = {"F": f_val, "p": p_val}
        results["anova_per_dimension"] = anova_results

        #wrong...
        # y_reshaped = y.reshape(-1, 1)
        # caa = CCA(n_components=1)
        # X_caa, y_caa = caa.fit_transform(X, y_reshaped)

        # canonical_corr = np.corrcoef(X_caa[:, 0], y_caa.flatten())[0, 1]
        # results["cca_correlation"] = canonical_corr

        with open(json_file, "w") as f:
            json.dump(
                {"score_aggregated": score_aggregated, "results": results}, f, indent=4
            )

        return score_aggregated, results
