import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
import seaborn as sns
import os
import numpy as np

from gradiend.evaluation.xai.io import all_interesting_classes, load_gradiends_by_configs, Gender, Case, \
    GradiendGenderCaseConfiguration, article_mapping, pretty_dataset_mapping, pretty_model_mapping
from gradiend.util import init_matplotlib


def postprocess_encoded_values(df):
    # encoded is str of list of length 1, convert to float
    df['encoded'] = df['encoded'].apply(lambda x: float(x.strip('[]')))
    return df

def read_encoded_values(model, articles, highlighted_datasets, group_neutral_datasets=True, group_by_article_transition=True, group_neutral_datasets_by_articles=True):
    df_path = f'{model}_params_spl_test.csv'
    df_all_path = f'{model}_params_inv_True_spl_test_var_all.csv'

    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
    else:
        raise FileNotFoundError(f'Encoded values file not found: {df_path}')

    if os.path.exists(df_all_path):
        df_all = pd.read_csv(df_all_path)
    else:
        raise FileNotFoundError(f'Encoded values file not found: {df_all_path}')

    df, df_all = postprocess_encoded_values(df), postprocess_encoded_values(df_all)
    df['type'] = 'df'
    df['type'] = 'df_all'
    df_all = df_all[df_all['label'].isin({'der', 'die', 'das', 'den', 'dem', 'des'})]

    df_all_dataset_sizes = df_all.groupby('dataset_labels').size()
    min_size = int(df_all_dataset_sizes.min())
    df_all = df_all.groupby('dataset_labels').apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)

    if articles:
        df_all = df_all[df_all['mask_labels'].isin(articles) & df_all['label'].isin(articles)]
        if group_by_article_transition:
            df_all['dataset_labels'] = '$' +  df_all['label'] + r'\!\to\!' + df_all['mask_labels'] + '$'
    else:
        df_all['dataset_labels'] = df_all['label']

    pretty_neutral = r'\dataNEUT'
    df['dataset_labels'] = df['dataset_labels'].fillna(pretty_neutral)
    df_neutral = df[df['dataset_labels'] == pretty_neutral]
    df_not_neutral = df[df['dataset_labels'] != pretty_neutral]
    df_neutral_dataset_sizes = df_neutral.groupby('dataset_labels').size()
    min_size = int(df_neutral_dataset_sizes.min())
    df_neutral = df_neutral.groupby('dataset_labels').apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)
    df = pd.concat([df_not_neutral, df_neutral], ignore_index=True)


    def map_dataset_labels(ds):
        if ds not in article_mapping:
            return ds

        article_source = article_mapping[ds]
        if ds in highlighted_datasets:
            other_highlighted_datasets = [d for d in highlighted_datasets if d != ds]
            if len(other_highlighted_datasets) > 1:
                raise ValueError("More than one highlighted dataset found for mapping.")
            article_target = article_mapping[other_highlighted_datasets[0]]
            pretty_ds = pretty_dataset_mapping[ds]
            return rf'${pretty_ds}\ ({article_source}\!\to\!{article_target})$'
        else:
            article_target = article_source

        if group_neutral_datasets_by_articles:
            return rf'{article_source} $\to$ {article_target}'

        return rf'{ds} ({article_source} $\to$ {article_target})'

    df['dataset_labels'] = df['dataset_labels'].apply(map_dataset_labels)
    highlighted_datasets = [map_dataset_labels(d) for d in highlighted_datasets]


    if group_neutral_datasets:
        other_datasets = [d for d in df['dataset_labels'].unique().tolist() if d not in highlighted_datasets + ['Neutral']]
        df.loc[df['dataset_labels'].isin(other_datasets), 'dataset_labels'] = 'Factual Neutral'

    sizes = df_all.groupby('dataset_labels').size()
    if sizes.empty:
        raise ValueError("No valid dataset_labels found for balancing.")

    min_size = int(sizes.min())

    print('Balancing datasets to size:', min_size)
    df_all = df_all.groupby('dataset_labels').apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)
    #return df
    final_df =  pd.concat([df, df_all], ignore_index=True)
    #final_df = df.copy()

    # sort columns
    datasets_present = final_df['dataset_labels'].unique().tolist()
    other_article_transitions = df_all['dataset_labels'].unique().tolist()
    # sort datasets, start with datasets_in_main_df, then alphabetical for the rest, end with Neutral
    remaining_datasets = sorted([d for d in datasets_present if d not in highlighted_datasets + other_article_transitions and 'neut' not in d.lower()])

    mean_encoded = final_df.groupby('dataset_labels')['encoded'].mean().to_dict()
    highlighted_datasets_sorted = sorted(highlighted_datasets, key=lambda x: mean_encoded.get(x, 0), reverse=True)
    other_article_transitions_sorted = sorted(other_article_transitions, key=lambda x: mean_encoded.get(x, 0), reverse=True)
    mean_encoded_remaining = {d: mean_encoded.get(d, 0) for d in remaining_datasets}
    remaining_datasets_sorted = sorted(remaining_datasets)

    if group_neutral_datasets:
        neutrals = ["Factual Neutral", pretty_neutral]
    else:
        neutrals = [pretty_neutral]
    sorted_columns = highlighted_datasets_sorted + other_article_transitions_sorted + remaining_datasets_sorted + neutrals
    final_df['dataset_labels'] = pd.Categorical(final_df['dataset_labels'], categories=sorted_columns, ordered=True)
    final_df = final_df.sort_values('dataset_labels').reset_index(drop=True)
    return final_df

def plot_encoded_value_distribution(
        *models,
        model_names=None,
        highlighted_datasets=None,
        articles=None,
        group_neutral_datasets=True,
        group_neutral_datasets_by_articles=True,
        group_by_article_transition=True,
        pretty_gradiend="",
        base_model_id="",
):
    # read all the encoded values data
    processed_dfs = []
    plot_model_names = []

    # Loop through each model and prepare the data
    for i, model in enumerate(models):
        # Read encoded values for this model
        df = read_encoded_values(model, articles, highlighted_datasets=highlighted_datasets, group_neutral_datasets=group_neutral_datasets, group_by_article_transition=group_by_article_transition, group_neutral_datasets_by_articles=group_neutral_datasets_by_articles)

        # Add a column to identify the model
        if model_names:
            df['model'] = model_names[i]
            plot_model_names.append(model_names[i].replace('\\', ''))
        else:
            model_base_name = model.replace('results/experiments/gradiend/', '').removesuffix('/0').replace('/dim_1_inv_gradient', '').replace('\\', '').replace('/', '_')
            plot_model_names.append(model_base_name)
            df['model'] = model_base_name

        # Collect the dataframe with the model label
        processed_dfs.append(df)

    font_size = 20

    # Concatenate all the dataframes into one, so we can plot them together
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    # Initialize the plot
    plt.figure(figsize=(9, 2.5))

    combined_df['type2'] = combined_df['dataset_labels']
    combined_df.loc[combined_df['dataset_labels'] == None, 'type2'] = "Neutral"
    combined_df['renamed_type'] = combined_df['type2'] #.map(rename_type_dict)

    combined_df = combined_df[combined_df['state'] != 'B'].reset_index(drop=True)

    paired = get_cmap("Paired")

    colors = paired.colors  # list of RGBA tuples

    custom_colors = [colors[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

    models = combined_df['model'].unique().tolist()
    if len(models) > 1:
        raise ValueError()

    combined_df['model'] = pretty_gradiend

    # Plot the distribution of the "encoded_value" column
    # Grouped by "type" and separated by "model" using hue
    sns.violinplot(x='model',
                   y='encoded',
                   hue='renamed_type',
                   data=combined_df,
                   split=True,
                   inner='quartile',
                   palette=custom_colors,
                   density_norm='width',
                   linewidth=0.7,
                   zorder=5,
                   cut=0,
                   )

    # Customize the plot
    plt.xlabel('', fontsize=1)
    plt.ylabel('$h$', fontsize=font_size)
    yticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    plt.xticks(fontsize=font_size-4)
    plt.yticks(yticks, fontsize=font_size-4)

    # make legend horizontal
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05), title_fontsize='large', fontsize=font_size-10, ncol=6)
    total_name = base_model_id + '__' + '_'.join(plot_model_names)

    # Show the plot
    plt.tight_layout()
    plt.grid(zorder=0)
    output = f'img/encoded_values_ddd_{total_name}{"_group_transitions" if group_by_article_transition else ""}{"_group_neutral" if group_neutral_datasets else ""}{"_group_neutral_by_articles" if group_neutral_datasets_by_articles else ""}.pdf'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.show()





def plot_encoded_value_distribution_stacked_models(
        model_dirs,
        model_ids,
        highlighted_datasets=None,
        articles=None,
        group_neutral_datasets=True,
        group_neutral_datasets_by_articles=True,
        group_by_article_transition=True,
        output=None,
):
    assert len(model_dirs) == len(model_ids)

    processed = []

    for model_dir, model_id in zip(model_dirs, model_ids):
        df = read_encoded_values(
            model_dir,
            articles,
            highlighted_datasets=highlighted_datasets,
            group_neutral_datasets=group_neutral_datasets,
            group_by_article_transition=group_by_article_transition,
            group_neutral_datasets_by_articles=group_neutral_datasets_by_articles,
        )

        df = df[df["state"] != "B"].reset_index(drop=True)

        df["model"] = model_id
        df["type2"] = df["dataset_labels"]
        df.loc[df["dataset_labels"].isna(), "type2"] = r"\dataNEUT"
        df["renamed_type"] = df["type2"]

        processed.append(df)

    combined_df = pd.concat(processed, ignore_index=True)

    paired = get_cmap("Paired")
    custom_colors = [paired.colors[i] for i in range(12)]

    font_size = 15
    n_models = len(model_ids)

    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(9, 1.5 * n_models),
        sharey=True,
    )

    if n_models == 1:
        axes = [axes]

    legend_handles = legend_labels = None
    y_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]

    for ax, model_id in zip(axes, model_ids):
        sub_df = combined_df[combined_df["model"] == model_id].copy()

        # dummy x so violins stay vertical
        sub_df["x"] = ""

        sns.violinplot(
            x="x",
            y="encoded",
            hue="renamed_type",
            data=sub_df,
            split=True,
            inner="quartile",
            palette=custom_colors,
            density_norm="width",
            linewidth=0.7,
            cut=0,
            ax=ax,
            legend=False,      # <<< CRITICAL
            zorder=5,
        )

        ax.set_xlabel("")
        pretty_model_id = pretty_model_mapping[model_id]
        ax.set_ylabel(f"{pretty_model_id}\n$h$", fontsize=font_size)
        ax.tick_params(axis="y", labelsize=font_size - 4)
        ax.set_xticks([])
        ax.set_yticks(y_ticks)

        ax.grid(zorder=0)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()


        # print mean encoded value by renamed_type
        mean_encoded = sub_df.groupby("renamed_type")["encoded"].mean()
        print(f"Mean encoded values for model {model_id}:")
        for renamed_type, mean_value in mean_encoded.items():
            print(f"  {renamed_type}: {mean_value:.4f}")

    from matplotlib.patches import Patch

    # AFTER the plotting loop
    hue_levels = combined_df["renamed_type"].unique().tolist()

    paired = plt.get_cmap("Paired")
    custom_colors = [paired(i) for i in range(len(hue_levels))]

    legend_handles = [
        Patch(facecolor=custom_colors[i], edgecolor="black", label=lbl)
        for i, lbl in enumerate(hue_levels)
    ]

    fig.legend(
        handles=legend_handles,
        labels=hue_levels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=6,
        fontsize=font_size - 4.5,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output is None:
        name = "_".join(model_ids)
        output = (
            f"img/encoded_values_stacked_models/{name}"
            f'{"_group_transitions" if group_by_article_transition else ""}'
            f'{"_group_neutral" if group_neutral_datasets else ""}'
            f'{"_group_neutral_by_articles" if group_neutral_datasets_by_articles else ""}.pdf'
        )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.show()


def plot_encoded_value_distribution_caller():
    init_matplotlib(use_tex=True)

    model_ids = [
        "bert-base-german-cased",
        "gbert-large",
        "ModernGBERT_1B",
        "EuroBERT-210m",
        "german-gpt2",
        "Llama-3.2-3B",
    ]

    # group by configuration, stack models
    for articles, config in all_interesting_classes.items():
        for c in config:

            try:
                # compute highlighted datasets ONCE per config (independent of model)
                cfg0 = GradiendGenderCaseConfiguration(*c, model_id=model_ids[0])
                highlighted = cfg0.datasets

                model_dirs = []
                for model_id in model_ids:
                    cfg = GradiendGenderCaseConfiguration(*c, model_id=model_id)
                    model_dirs.append(cfg.gradiend_dir)


                output_file = f"img/encoded_values_stacked_models/{articles[0]}_{articles[1]}_{c[0]}_{c[1]}_{c[2]}.pdf"
                plot_encoded_value_distribution_stacked_models(
                    model_dirs=model_dirs,
                    model_ids=model_ids,
                    highlighted_datasets=highlighted,
                    articles=articles,
                    group_neutral_datasets=False,
                    group_neutral_datasets_by_articles=True,
                    group_by_article_transition=True,
                    output=output_file,
                )
            except NotImplementedError as e:
                print(e)

if __name__ == '__main__':
    plot_encoded_value_distribution_caller()