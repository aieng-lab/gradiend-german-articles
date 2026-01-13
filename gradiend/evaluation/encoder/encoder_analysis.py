from abc import ABC, abstractmethod
from itertools import combinations
import os

from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns

import pandas as pd
import torch

from gradiend.data.util import (
    get_default_prediction_file_name,
    get_file_name,
    json_dumps,
    json_loads,
)


def z_score(x, groupby=None, key=None):
    if isinstance(x, pd.DataFrame):
        assert key is not None

        if groupby is not None:
            result = x.groupby(groupby).apply(lambda x: z_score(x[key]))
            return result.reset_index(level=0, drop=True)
        x = x[key]

    mean = x.mean()
    std = x.std()
    return (x - mean) / std


def get_correlation(df, method):
    if method == "pearson":
        corr, p_value = stats.pearsonr(df["state_value"], df["encoded"])
    elif method == "spearman":
        corr, p_value = stats.spearmanr(df["state_value"], df["encoded"])
    else:
        raise ValueError(f"Unknown method: {method}")

    return {"correlation": corr, "p_value": p_value}


def get_pearson_correlation(df):
    return get_correlation(df, method="pearson")


def get_spearman_correlation(df):
    return get_correlation(df, method="spearman")


class EncoderAnalysis(ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gender_keys = list(config["categories"].keys())
        self.articles = list(
            {
                article
                for key in self.gender_keys
                for article in config["categories"][key]["articles"]
            }.union(set(self.config.get('articles', [])))
        )
        self.det_combination = config["plot_name"]

    @abstractmethod
    def analyse_encoder(self, model_with_gradiend, dataset, output, plot=False):
        pass

    @abstractmethod
    def get_model_metrics(
        self, *encoded_values, config, prefix=None, suffix=".csv", **kwargs
    ):
        pass

    def read_default_predictions(self, model, combinations=None):
        file = get_default_prediction_file_name(model, combinations)

        try:
            cache_default_predictions = pd.read_csv(file)
            cache_default_predictions.set_index("text", inplace=True)
            for i in self.articles:
                cache_default_predictions[i] = cache_default_predictions[i].apply(
                    json_loads
                )

            cache_default_predictions["most_likely_token"] = cache_default_predictions[
                "most_likely_token"
            ].apply(json_loads)
            cache_default_predictions["label"] = cache_default_predictions[
                "label"
            ].apply(json_loads)
            cache_default_predictions_dict = cache_default_predictions.to_dict(
                orient="index"
            )
        except Exception: # todo make the file depend on articles
            cache_default_predictions_dict = {}

        return cache_default_predictions_dict

    def write_default_predictions(self, default_predictions, model):
        file = get_default_prediction_file_name(model)
        # Ensure the directory exists
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        cache_default_predictions = pd.DataFrame.from_dict(
            default_predictions, orient="index"
        )

        for article in self.articles:
            cache_default_predictions[article] = cache_default_predictions[
                article
            ].apply(json_dumps)

        cache_default_predictions["most_likely_token"] = cache_default_predictions[
            "most_likely_token"
        ].apply(json_dumps)
        cache_default_predictions["label"] = cache_default_predictions["label"].apply(
            json_dumps
        )
        cache_default_predictions.reset_index(inplace=True)
        cache_default_predictions.rename(columns={"index": "text"}, inplace=True)
        cache_default_predictions.to_csv(file, index=False)

    def evaluate_determiners(self, model, tokenizer, masked_text):
        """
        Evaluate the model on masked language modeling (MLM) task. Specifically, determine the probabilities of the german determiner tokens.

        Args:
        - model: The model (e.g. BertForMaskedLM).
        - tokenizer: The tokenizer.
        - masked_text: The text with a masked token.
        """
        # Tokenize the input text
        inputs = tokenizer(masked_text, return_tensors="pt")
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get the index of the masked token
        is_generative = tokenizer.mask_token_id is None
        if is_generative:
            mask_token_index = len(inputs["input_ids"]) - 1
        else:
            mask_token_index = torch.where(
                inputs["input_ids"] == tokenizer.mask_token_id
            )[1]

        # Pass the inputs through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the logits and softmax to get probabilities
        logits = outputs.logits
        is_decoder_mlm_head = len(logits.shape) == 2

        if is_decoder_mlm_head:
            mask_token_logits = logits[0, :]
        else:
            mask_token_logits = logits[0, mask_token_index, :]
        probabilities = torch.softmax(mask_token_logits, dim=-1)

        token_ids = {}

        if is_decoder_mlm_head:
            target_tokens = tokenizer.convert_ids_to_tokens(model.target_token_ids)
            for article in self.articles:
                token_ids[article] = target_tokens.index(article)
        else:
            for article in self.articles:
                token_ids[article] = tokenizer.convert_tokens_to_ids(article)


        # Get the probabilities for the determiners.
        shape = probabilities.shape

        if shape[0] == 0:
            result = {
                **{article: [] for article in self.articles},
                "most_likely_token": [],
            }
        else:
            if is_decoder_mlm_head:
                probabilities = probabilities.unsqueeze(0)

            token_probabilities = {}

            for key, token_id in token_ids.items():
                token_probabilities[key] = probabilities[:, token_id].tolist()

            # Determine the most likely token
            most_likely_token_id = torch.argmax(probabilities, dim=1).tolist()

            # Determine which token was the most likely
            most_likely_token = [tokenizer.decode(id) for id in most_likely_token_id]

            if shape[0] == 1:
                for key, probability in token_probabilities.items():
                    token_probabilities[key] = probability[0]

                most_likely_token = most_likely_token[0]

            # Prepare the result dictionary
            result = {**token_probabilities, "most_likely_token": most_likely_token}

        return result

    def read_article_encoded_values(self, file):
        encoded_values = get_file_name(
            file, file_format="csv", max_size=None, split="test"
        )
        df_encoded = pd.read_csv(encoded_values)

        for article in self.articles:
            df_encoded[article] = df_encoded[article].apply(json_loads)

        df_encoded["labels"] = df_encoded["labels"].apply(json_loads)
        df_encoded["most_likely_token"] = df_encoded["most_likely_token"].apply(
            json_loads
        )

        return df_encoded

    def plot_model_results(self, encoded_values, title="", y="z_score"):
        if isinstance(encoded_values, str):
            results = self.read_article_encoded_values(encoded_values)
            if not title:
                title = encoded_values.removesuffix(".csv")
        else:
            results = encoded_values

        plot_results = results.copy()
        plot_results["plot_state"] = plot_results["type"]
        if y == "encoded":
            plot_results.loc[
                plot_results["type"] == f"{self.det_combination} masked", "plot_state"
            ] = plot_results["state"]
        else:
            plot_results = plot_results[
                plot_results["type"] == f"{self.det_combination} masked"
            ].reset_index(drop=True)
            plot_results["plot_state"] = plot_results["state"]

        #    plot_results = results[results['type'] == 'gender masked'].sort_values(by='state').reset_index(drop=True)
        sns.boxplot(x="plot_state", y=y, data=plot_results)
        plt.grid()
        plt.title(title)
        file = f"results/img/z_score/{title}_{y}.png"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file)
        plt.show()

    def get_std_stats(self, df):

        def map_category(label):
            for key in self.gender_keys:
                if label in self.config["categories"][key]["labels"]:
                    return key
            return None

        df["category"] = df["dataset_labels"].map(map_category)

        std_keys = {
            f"std_{key}": df[df["category"] == key]["encoded"].std()
            for key in self.gender_keys
        }
        std_diffs = {
            f"std_diff_{k1}_{k2}": abs(std_keys[k2] - std_keys[k1])
            for k1, k2 in combinations(std_keys, 2)
        }

        return {
            **std_keys,
            **std_diffs,
        }
