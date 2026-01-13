import itertools
import json
import os.path

from scipy.stats import stats
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

import pprint

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from omegaconf import open_dict

from gradiend.evaluation.encoder.de_encoder_analysis import DeEncoderAnalysis
from gradiend.combined_models.combined_gradiends import CombinedEncoderDecoder
from gradiend.model import ModelWithGradiend
from gradiend.data import read_article_ds, read_geneutral, read_gender_data, get_gender_words, \
    write_default_predictions, read_default_predictions, json_dumps, json_loads, read_genter, read_namexact, \
    read_de_neutral
from gradiend.data import read_names_data
import seaborn as sns


from gradiend.util import token_distance, get_files_and_folders_with_prefix, evaluate_he_she, find_outliers, \
    case_gender_mapping


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



def plot_model_results(encoded_values, title='', y='z_score'):
    if isinstance(encoded_values, str):
        results = read_encoded_values(encoded_values)
        if not title:
            title = encoded_values.removesuffix('.csv')
    else:
        results = encoded_values

    plot_results = results.copy()
    plot_results['plot_state'] = plot_results['type']
    if y == 'encoded':
        plot_results.loc[plot_results['type'] == 'gender masked', 'plot_state'] = plot_results['state']
    else:
        plot_results = plot_results[plot_results['type'] == 'gender masked'].reset_index(drop=True)
        plot_results['plot_state'] = plot_results['state']

#    plot_results = results[results['type'] == 'gender masked'].sort_values(by='state').reset_index(drop=True)
    sns.boxplot(x='plot_state', y=y, data=plot_results)
    plt.grid()
    plt.title(title)
    file = f'results/img/z_score/{title}_{y}.png'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    plt.savefig(file)
    plt.show()



def analyze_model(model_with_gradiend, genter_df, names_df, output, df_no_gender=None, include_B=False, plot=False):
    if not include_B and 'B' in names_df['gender'].unique():
        if not 'genders' in names_df:
            raise ValueError('The names_df must contain the key genders if include_B is False')
        names_df = names_df[names_df['genders'] != 'B']

    model = model_with_gradiend.base_model
    tokenizer = model_with_gradiend.tokenizer
    mask_token = tokenizer.mask_token
    is_generative = model_with_gradiend.is_generative

    cache_default_predictions_dict = read_default_predictions(model)

    modified_cache = []

    def get_default_prediction(masked_text):
        if masked_text in cache_default_predictions_dict:
            return cache_default_predictions_dict[masked_text]

        # calculate the predictions
        predictions = evaluate_he_she(model, tokenizer, masked_text)
        cache_default_predictions_dict[masked_text] = predictions
        if not modified_cache:
            modified_cache.append(True)
        return predictions

    source = model_with_gradiend.gradiend.kwargs['training']['source']

    male_names = itertools.cycle(names_df[names_df['gender'] == 'M'].iterrows())
    female_names = itertools.cycle(names_df[names_df['gender'] == 'F'].iterrows())

    filled_texts = []

    def process_entry(row, plot=False):
        if is_generative:
            masked = row['masked'].split('[PRONOUN]')[0]
        else:
            masked = row['masked'].replace('[PRONOUN]', mask_token)
        encoded_values = []
        genders = []
        names = []
        token_distances = []
        labels = []
        default_predictions = {k: [] for k in ['he', 'she', 'most_likely_token', 'label']}

        masked_token_distance = token_distance(tokenizer, masked, '[NAME]', mask_token)

        # todo make the encoding batched!

        inputs = []
        for i, entry in [next(female_names), next(male_names)]:
            name = entry['name']
            filled_text = masked.replace('[NAME]', name)
            filled_texts.append(filled_text)
            label = 'he' if entry['gender'] == 'M' else 'she'

            if source == 'diff':
                label_factual = 'he' if label == 'M' else 'she'
                label_counter_factual = 'she' if label == 'M' else 'he'
                inputs_factual = model_with_gradiend.create_inputs(filled_text, label_factual)
                grads_factual = model_with_gradiend.forward_pass(inputs_factual, return_dict=False)
                inputs_counter_factual = model_with_gradiend.create_inputs(filled_text, label_counter_factual)
                grads_counter_factual = model_with_gradiend.forward_pass(inputs_counter_factual, return_dict=False)
                grads = grads_factual - grads_counter_factual
                inputs.append(grads)
                encoded = model_with_gradiend.gradiend.encoder(grads).item()
            else:
                if source == 'gradient':
                    masked_label = label
                elif source == 'inv_gradient':
                    masked_label = 'he' if label == 'she' else 'she'
                else:
                    raise ValueError(f'Unknown source: {source}')
                inputs.append((filled_text, masked_label))
                encoded = model_with_gradiend.encode(filled_text, label=masked_label)

            encoded_values.append(encoded)
            genders.append(entry['gender'])
            names.append(name)
            token_distances.append(masked_token_distance)
            labels.append([label] * row['pronoun_count'])

            default_prediction = get_default_prediction(filled_text)
            default_prediction['label'] = label
            for key, value in default_prediction.items():
                default_predictions[key].append(value)

        results = pd.DataFrame({
            'text': masked,
            'name': names,
            'state': genders,
            'encoded': encoded_values,
            'token_distance': token_distances,
            'labels': labels,
            'type': 'gender masked',
            **default_predictions,
        })

        results['state_value'] = results['state'].map({'M': 0, 'F': 1, 'B': 0.5})
        results['z_score'] = z_score(results['encoded'])
        results = results.sort_values(by='state')

        if plot:
            plt.title(row['masked'])
            sns.boxplot(x='state', y='z_score', data=results)
            plt.show()

        return results

        # todo correlate with names that might be used for the other gender!!!

    # Enable the progress_apply method for DataFrames
    tqdm.pandas(desc="Analyze with GENTER Test Data")
    genter_df['genter'] = genter_df.progress_apply(process_entry, axis=1)
    results = genter_df['genter'].tolist()

    gender_tokens = get_gender_words(tokenizer=tokenizer)
    torch.manual_seed(42)
    if df_no_gender is not None:
        if len(df_no_gender) > 10000:
            df_no_gender = df_no_gender.head(10000).reset_index(drop=True)
        texts = []
        encoded_values = []
        labels = []
        default_predictions = {k: [] for k in ['he', 'she', 'most_likely_token', 'label']}

        for i, row in tqdm(list(df_no_gender.iterrows()), desc='No gender data'):
            text = row['text']
            encoded, masked_text, label = model_with_gradiend.mask_and_encode(text, ignore_tokens=gender_tokens, return_masked_text=True, single_mask=True)
            texts.append(masked_text)
            encoded_values.append(encoded)
            labels.append(label)

            default_prediction = get_default_prediction(masked_text)
            default_prediction['label'] = label
            for key, value in default_prediction.items():
                default_predictions[key].append(value)

        result = pd.DataFrame({
            'text': texts,
            'name': None,
            'state': None,
            'encoded': encoded_values,
            'token_distance': None,
            'type': 'no gender',
            **default_predictions,
        })
        results.append(result)

    texts = []
    encoded_values = []
    labels = []
    default_predictions = {k: [] for k in ['he', 'she', 'most_likely_token', 'label']}
    torch.manual_seed(42)
    for text in tqdm(filled_texts, desc='GENTER data without gender words masked'):
        encoded, masked_text, label = model_with_gradiend.mask_and_encode(text, ignore_tokens=gender_tokens, return_masked_text=True)
        texts.append(text)
        encoded_values.append(encoded)
        labels.append(label)

        default_prediction = get_default_prediction(masked_text)
        default_prediction['label'] = label
        for key, value in default_prediction.items():
            default_predictions[key].append(value)

    result = pd.DataFrame({
        'text': texts,
        'name': None,
        'state': None,
        'encoded': encoded_values,
        'token_distance': None,
        'type': 'no gender masked',
        **default_predictions,
    })
    results.append(result)

    if modified_cache:
        write_default_predictions(cache_default_predictions_dict, model)

    total_results = pd.concat(results)

    mean = total_results['encoded'].mean()
    std = total_results['encoded'].std()
    total_results['global_z_score'] = (total_results['encoded'] - mean) / std


    total_results['he'] = total_results['he'].apply(json_dumps)
    total_results['she'] = total_results['she'].apply(json_dumps)
    total_results['label'] = total_results['label'].apply(json_dumps)
    total_results['most_likely_token'] = total_results['most_likely_token'].apply(json_dumps)

    total_results.to_csv(output, index=False)



    if plot:
        # plot results
        plot_model_results(total_results, title=output.removesuffix('.csv'))

        plot_results = total_results[total_results['type'] == 'gender masked'].sort_values(by='state').reset_index(drop=True)
        sns.boxplot(x='state', y='encoded', data=plot_results)
        plt.title(model_with_gradiend.name_or_path)
        plt.show()

        cor = np.nanmean([text_df[['encoded', 'state_value']].corr(method='pearson')['encoded']['state_value'] for text, text_df in plot_results.groupby('text')])
        print('Correlation', cor)

        # now without B
        plot_results_MF = plot_results[plot_results['state'] != 'B']
        cor = np.nanmean([text_df[['encoded', 'state_value']].corr(method='pearson')['encoded']['state_value'] for text, text_df in plot_results_MF.groupby('text')])
        print('Correlation MF', cor)

    return total_results



def get_correlation(df, method):
    if method == 'pearson':
        corr, p_value = stats.pearsonr(df['state_value'], df['encoded'])
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(df['state_value'], df['encoded'])
    else:
        raise ValueError(f'Unknown method: {method}')

    return {'correlation': corr, 'p_value': p_value}


def get_pearson_correlation(df):
    return get_correlation(df, method='pearson')

def get_spearman_correlation(df):
    return get_correlation(df, method='spearman')

def bhattacharyya_coefficient(p, q):
    return np.sum(np.sqrt(p * q))

def js_divergence(p, q):
    return jensenshannon(p, q)**2


def get_std_stats(df):
    std_M = df[df['state'] == 'M']['encoded'].std()
    std_F = df[df['state'] == 'F']['encoded'].std()
    std_diff = abs(std_M - std_F)

    return {
        'std_M': std_M,
        'std_F': std_F,
        'std_diff': std_diff,
    }


def read_encoded_values(file):
    encoded_values = get_file_name(file, file_format='csv', max_size=None, split='test')
    df_encoded = pd.read_csv(encoded_values)

    df_encoded['he'] = df_encoded['he'].apply(json_loads)
    df_encoded['she'] = df_encoded['she'].apply(json_loads)
    df_encoded['labels'] = df_encoded['labels'].apply(json_loads)
    df_encoded['most_likely_token'] = df_encoded['most_likely_token'].apply(json_loads)

    return df_encoded

def read_article_encoded_values(file,config):
    encoded_values = get_file_name(file, file_format='csv', max_size=None, split='test')
    df_encoded = pd.read_csv(encoded_values)

    for article in config['articles']: 
        df_encoded[article] = df_encoded[article].apply(json_loads)  
    df_encoded['labels'] = df_encoded['labels'].apply(json_loads)
    df_encoded['most_likely_token'] = df_encoded['most_likely_token'].apply(json_loads)

    return df_encoded


def get_file_name(base_file_name, file_format=None, **kwargs):
    base_name = os.path.basename(base_file_name)
    output = str(base_file_name)
    if '.' in base_name[-5:]:
        current_file_format = base_name.split('.')[-1]
        if current_file_format in {'csv', 'json', 'txt', 'tsv'}:
            if current_file_format != file_format:
                raise ValueError(f'Provided format of file {current_file_format} does not match key word argument {file_format}: {base_file_name}')
            output = base_file_name[:-len(file_format) - 1]

    first_param = True
    for key, value in sorted(kwargs.items()):
        if value is not None:
            if first_param:
                output += '_params'
                first_param = False

            output += f'_{key[:3]}_{value}'

    if file_format and not output.endswith(file_format):
        output += '.' + file_format

    return output


def get_model_metrics(*encoded_values, prefix=None, suffix='.csv', **kwargs):
    if prefix:
        # find all models in the folder with the suffix
        encoded_values = list(encoded_values) + get_files_and_folders_with_prefix(prefix, suffix=suffix)

    if len(encoded_values) > 1:
        metrics = {}
        for ev in encoded_values:
            m = get_model_metrics(ev, **kwargs)
            metrics[ev] = m

        return metrics

    raw_encoded_values = encoded_values[0]

    encoded_values = get_file_name(raw_encoded_values, file_format='csv', **kwargs)
    json_file = encoded_values.replace('.csv', '.json')

    try:
        return json.load(open(json_file, 'r+'))
    except FileNotFoundError:
        print('Computing model metrics for', encoded_values)

    df_all = pd.read_csv(encoded_values)
    try:
        df = df_all[df_all['type'] == 'gender masked']
    except KeyError:
        df = df_all

    min_ds_size = df.groupby('ds').size().min()
    df = df.groupby('ds').apply(lambda group: group.sample(n=min_ds_size, random_state=42)).reset_index(drop=True)

    df_copy = df
    df_without_B = df[df['state'] != 'B'].copy()


    df_without_B['z_score_MF'] = z_score(df_without_B, key='encoded', groupby='text')
    df_without_B['global_z_score_MF'] = z_score(df_without_B['encoded'])

    # state_value in text_df still refers to old labeling with M being 1!
    acc_M_positive = np.mean([((text_df['encoded'] >= 0) == text_df['state_value'].astype(bool)).sum() / len(text_df) for text, text_df in df_without_B.groupby('text')])
    acc_M_negative = np.mean([((text_df['encoded'] < 0) == text_df['state_value'].astype(bool)).sum() / len(text_df) for text, text_df in df_without_B.groupby('text')])
    acc_optimized_border_M_pos = np.mean([max(((text_df['encoded'] < threshold) == text_df['state_value'].astype(bool)).sum() / len(text_df) for threshold in np.arange(-1.0, 1.0, 0.1)) for text, text_df in df_without_B.groupby('text')])
    acc_optimized_border_M_neg = np.mean([max(((text_df['encoded'] > threshold) == text_df['state_value'].astype(bool)).sum() / len(text_df) for threshold in np.arange(-1.0, 1.0, 0.1)) for text, text_df in df_without_B.groupby('text')])

    acc_M_positive_global = np.mean([((text_df['global_z_score_MF'] >= 0) == text_df['state_value'].astype(bool)).sum() / len(text_df) for text, text_df in df_without_B.groupby('text')])
    acc_M_negative_global = np.mean([((text_df['global_z_score_MF'] < 0) == text_df['state_value'].astype(bool)).sum() / len(text_df) for text, text_df in df_without_B.groupby('text')])

    def calculate_optimal_proportion(group):
        # Get the global_z_score_MF and state_value columns
        z_scores = group['global_z_score_MF'].values
        state_values = group['state_value'].astype(bool).values

        # Get all unique thresholds
        unique_thresholds = np.unique(z_scores)

        # Vectorize the thresholding operation
        proportions = np.array([(z_scores < threshold) == state_values for threshold in unique_thresholds])
        proportions = proportions.sum(axis=1) / len(group)

        # Return the maximum proportion
        return proportions.max()

    acc_optimized_border_M_pos_global = np.mean([max(((text_df['global_z_score_MF'] < threshold) == text_df['state_value'].astype(bool)).sum() / len(text_df) for threshold in np.arange(-1.0, 1.0, 0.1)) for text, text_df in df_without_B.groupby('text')])
    acc_optimized_border_M_neg_global = np.mean([max(((text_df['global_z_score_MF'] > threshold) == text_df['state_value'].astype(bool)).sum() / len(text_df) for threshold in np.arange(-1.0, 1.0, 0.1)) for text, text_df in df_without_B.groupby('text')])


    # rename the keys such that blanks are replaced by '_'
    df_all = df_all.rename(columns=lambda x: x.replace(' ', '_'))
    encoded_abs_means = df_all.groupby('type')['encoded'].apply(lambda group: group.abs().mean()).to_dict()
    encoded_means = df_all.groupby('type')['encoded'].apply(lambda group: group.mean()).to_dict()
    print(encoded_means)

    # map encoded values to the predicted class, i.e. >= 0.5 -> female, <= -0.5 -> male, >-0.5 & <0.5 -> neutral
    df_all['predicted_female_pos'] = df_all['encoded'].apply(lambda x: 1 if x >= 0.5 else (-1 if x <= -0.5 else 0))
    df_all['predicted_male_pos'] = df_all['encoded'].apply(lambda x: 1 if x <= -0.5 else (-1 if x >= 0.5 else 0))
    labels = df_all['state'].apply(lambda x: 1 if x == 'F' else (-1 if x == 'M' else 0))
    df_all['state_value'] = labels
    balanced_acc_female_pos = balanced_accuracy_score(df_all['predicted_female_pos'], labels)
    balanced_acc_male_pos = balanced_accuracy_score(df_all['predicted_male_pos'], labels)
    acc_total = max(balanced_acc_female_pos, balanced_acc_male_pos)

    pearson_text_cor = df.groupby('text').apply(get_pearson_correlation, include_groups=False)
    spearman_text_cor = df.groupby('text').apply(get_spearman_correlation, include_groups=False)
    pearson_text_MF_cor = df_without_B.groupby('text').apply(get_pearson_correlation, include_groups=False)
    spearman_text_MF_cor = df_without_B.groupby('text').apply(get_spearman_correlation, include_groups=False)

    pearson_total = get_pearson_correlation(df_all)
    spearman_total = get_spearman_correlation(df_all)

    pearson = get_pearson_correlation(df)
    spearman = get_spearman_correlation(df)

    pearson = get_pearson_correlation(df_without_B)
    spearman_MF = get_spearman_correlation(df_without_B)

    scores = {
        'pearson_total': pearson_total['correlation'],
        'pearson_total_p_value': pearson_total['p_value'],
        'spearman_total': spearman_total['correlation'],
        'spearman_total_p_value': spearman_total['p_value'],
        'acc_total': acc_total,

        'pearson': pearson['correlation'],
        'pearson_p_value': pearson['p_value'],
        'spearmann': spearman,
        'spearman_p_value': spearman['p_value'],

        'pearson': pearson['correlation'],
        'pearson_p_value': pearson['p_value'],
        'spearman_MF': spearman_MF['correlation'],
        'spearman_MF_p_value': spearman_MF['p_value'],

        'pearson_text': np.mean([p['correlation'] for  p in pearson_text_cor]).item(),
        'spearman_text': np.mean([p['correlation'] for  p in spearman_text_cor]).item(),

        'pearson_text_MF': np.mean([p['correlation'] for  p in pearson_text_MF_cor]).item(),
        'spearman_text_MF': np.mean([p['correlation'] for  p in spearman_text_MF_cor]).item(),

        'acc': max(acc_M_negative, acc_M_positive),
        'acc_zscore': max(acc_M_negative_global, acc_M_positive_global),
        'acc_optimized': max(acc_optimized_border_M_neg, acc_optimized_border_M_pos),
        'acc_optimized_zscore': max(acc_optimized_border_M_neg_global, acc_optimized_border_M_pos_global),

        'encoded_abs_means': encoded_abs_means,
        'encoded_means': encoded_means,

        **get_std_stats(df),
    }

    print(scores)



    with open(json_file, 'w') as f:
        json.dump(scores, f, indent=4)

    return scores

def plot_encoded_value_distribution(config, *models, model_names=None):
    # read all the encoded values data
    # for each group 'type' plot the distribution of the encoded values in a single plot
    # Initialize the plot
    # Initialize a list to collect all the processed data
    processed_dfs = []
    plot_model_names = []

    # Loop through each model and prepare the data
    for i, model in enumerate(models):
        # Read encoded values for this model
        df = read_article_encoded_values(model, config)

        # Add a column to identify the model
        if model_names:
            df['model'] = model_names[i]
            plot_model_names.append(model_names[i].replace('\\', ''))
        else:
            model_base_name = model.split('/')[-1].replace('\\', '')
            plot_model_names.append(model_base_name)
            df['model'] = model_base_name

        # Collect the dataframe with the model label
        processed_dfs.append(df)

    font_size = 20

    # Concatenate all the dataframes into one, so we can plot them together
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    # Initialize the plot
    plt.figure(figsize=(9, 4))
   # plt.figure(figsize=(13, 3.5))

    rename_type_dict = {
        f"{config['plot_name']} masked": f"{config['plot_name']}", # todo write genter with \textsc?
        f"no {config['plot_name']} masked": f"{config['plot_name']}_0", #TODO change this to zero or smth 
        #'no gender': r'\geneutral'
    }
    combined_df['renamed_type'] = combined_df['type'].map(rename_type_dict)

    #combined_df = combined_df[combined_df['state'] != 'B'].reset_index(drop=True)

    article_keys = list(config['categories'].keys())

    combined_df['category'] = combined_df['dataset_labels'].apply(lambda x: next((key for key in article_keys if x in config['categories'][key]['labels']), None))

    # Plot the distribution of the "encoded_value" column
    # Grouped by "type" and separated by "model" using hue
    sns.violinplot(x='model',
                   y='encoded',
                   hue='renamed_type',
                   data=combined_df,
                   split=False,
                   inner='quartile',
                   palette='YlGnBu',
                   hue_order=list(rename_type_dict.values()),
                  )

    #sns.violinplot(data=combined_df, x="model", y="encoded", hue="category", split=True, density_norm='area', width=1, dodge=False)


    # sns.catplot(
    # data=combined_df,
    #     x="model",
    #     y="encoded",
    #     hue="category",
    #     kind="violin",
    #     col="category",  
    #     sharey=False,
    # )

    fig, ax = plt.subplots()
    
    sorted_keys = sorted(article_keys, key=lambda k: config['categories'][k]['encoding'], reverse=True)


    # for key in article_keys: 
    #     sns.violinplot(
    #     data=combined_df[combined_df['category'] == key],
    #     x="model",         
    #     y="encoded",       
    #     hue="category",   
    #     inner="quartile",
    #     hue_order=sorted_keys,
    #     palette=config['palette'],
    #     ax=ax,
    #     alpha=0.1
    # )
    
    # Customize the plot
    #plt.title('Distribution of Encoded Values by Type and Model')
    #plt.xlabel('Model', fontsize=font_size)
    plt.ylabel('Encoded Value $h$', fontsize=font_size)
    plt.xticks(fontsize=font_size-4)
    plt.yticks(fontsize=font_size-4)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Category")

  

    # make legend horizontal
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), title_fontsize='large', fontsize=font_size-2, ncol=3)

    # Rotate x-axis labels for better readability
    if not model_names:
        plt.xticks(rotation=45, ha='right', fontsize=font_size)

    total_name = '_'.join(plot_model_names)

    # Show the plot
    plt.tight_layout()
    plt.grid()
    output = f'results/img/encoded_values_{total_name}.pdf'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.show()


def compare_neuron_parts(model, decoder_part='decoder'):
    bert_with_ae = ModelWithGradiend.from_pretrained(model)
    diffs = []
    encoder = []
    decoder = []

    for (name_encoder, params_encoder), (name_decoder, params_decoder) in zip(bert_with_ae.ae_named_parameters(part='encoder'), bert_with_ae.ae_named_parameters(part=decoder_part)):
        assert name_encoder == name_decoder

        diff = (params_encoder - params_decoder).abs().detach().flatten().tolist()
        diffs += diff
        encoder += params_encoder.abs().detach().flatten().tolist()
        decoder += params_decoder.abs().detach().flatten().tolist()

    # Plot histograms
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[0].hist(encoder, bins=20, color='blue', alpha=0.7)
    ax[0].set_title('Encoder Parameters')
    ax[0].set_xlabel('Value')
    ax[0].set_ylabel('Frequency')

    ax[1].hist(decoder, bins=20, color='green', alpha=0.7)
    ax[1].set_title('Decoder Parameters')
    ax[1].set_xlabel('Value')
    ax[1].set_ylabel('Frequency')

    # Sampling diffs for better visualization
    ax[2].hist(diffs, bins=20, color='red', alpha=0.7)
    ax[2].set_title('Difference Between Encoder and Decoder Parameters')
    ax[2].set_xlabel('Value')
    ax[2].set_ylabel('Frequency')

    plt.show()



def analyze_neurons(model, part='encoder', include_he_she=True):
    print(f'Analyzing the neurons for the auto encoder model {model} (part={part})')
    bert_with_ae = ModelWithGradiend.from_pretrained(model)
    average_weights = {}
    proportion = {}
    threshold = 0.1

    for name, params in bert_with_ae.ae_named_parameters(part=part):
        data = params.detach().cpu().abs().numpy()
        average_weights[name] = data.mean()
        proportion[name] = (data >= threshold).sum() / data.size


    sorted_layers = list(sorted(average_weights.items(), key=lambda x: -x[1]))
    print(f'Top 10 relevant layers of {model}')
    pprint.pprint(sorted_layers[:10])

    layer_map = {k: v for k, v in bert_with_ae.ae_named_parameters(part=part)}
    word_embeddings = layer_map['base_model.embeddings.word_embeddings.weight']
    token_means = word_embeddings.abs().mean(dim=1)
    tokens = [bert_with_ae.raw_tokenizer.decode(index) for index in range(word_embeddings.shape[0])]
    token_means = {token: mean.cpu().item() for token, mean in zip(tokens, token_means) if not (token.startswith('[') and token.endswith(']'))}
    names_df = read_names_data(filter_excluded_words=True, minimum_count=0, max_entries=None)
    training_names_df = read_namexact(split='train')
    training_names = set(training_names_df['name'].str.lower().values)
    non_training_names = names_df[names_df['split'] != 'train']
    from gradiend.data import excluded_names
    all_names = set(names_df['name'].str.lower().values).union(excluded_names)
    names_used_during_training = {token: mean for token, mean in token_means.items() if token.lower() in training_names}
    all_names_not_used_during_training = {token: mean for token, mean in token_means.items() if token.lower() not in training_names and token.lower() in all_names}
    #names_count_1000_not_used_during_training = {token: mean for token, mean in token_means.items() if token.lower() not in training_names and token.lower() in all_names and names_df[names_df['name'].str.lower() == token.lower()]['count'].sum() > 1000}
    frequent_threshold = 100000

    frequent_names_not_used_during_training = set(name for name in names_df[names_df['count'] > frequent_threshold]['name'].str.lower() if name not in training_names)
    frequent_names_not_used_during_training = {token: mean for token, mean in token_means.items() if token.lower() in frequent_names_not_used_during_training}
    unfrequent_names_not_used_during_training = set(name for name in names_df[(names_df['count'] < frequent_threshold) & (names_df['count'] > 1000)]['name'].str.lower() if name not in training_names)
    unfrequent_names_not_used_during_training = {token: mean for token, mean in token_means.items() if token.lower() in unfrequent_names_not_used_during_training}

    gender_words = {word.lower() for word in read_gender_data(as_dict=False, include_gender_pronouns=True)}
    if not include_he_she:
        gender_words = {k for k in gender_words if k not in {'he', 'she'}}
    gender_words = {token: mean for token, mean in token_means.items() if token.lower() in gender_words}
    top_gender_word = find_outliers(gender_words, threshold=2, top_k=1)
    print('Top gender word outlier', top_gender_word)


    excluded_names = set(name.lower() for name in excluded_names)
    multiple_meaning_names = {token: mean for token, mean in token_means.items() if token.lower() in excluded_names}
    top_multiple_meaning_names = find_outliers(multiple_meaning_names, top_k=2, threshold=2)

    all_considered_tokens_so_far = set().union(names_used_during_training, frequent_names_not_used_during_training, unfrequent_names_not_used_during_training, multiple_meaning_names, gender_words)
    other_tokens = {token: mean for token, mean in token_means.items() if token.lower() not in all_considered_tokens_so_far}
    if not include_he_she:
        other_tokens = {token: mean for token, mean in other_tokens.items() if token not in {'he', 'she'}}

    other_token_outliers = find_outliers(other_tokens)
    top_outliers = find_outliers(other_token_outliers, top_k=3, threshold=1)
    # includes
    #   - gender biased professions (painter, sheriff, killer, ...)
    #   - words for humans
    #       - gender specific: papa, mum, grandma, mr, ...
    #       - non gender specific: children, family, people, kids, ...
    #   - pronouns: it, his, her, she, you, they, their, we,
    #   - colors: red, black, blue
    #   - endings
    #       - ##y (male name ending Billy, Johnny),
    #       - ##ie (female name ending Maggie, Katie),
    #       - ##lle (female name ending Michelle, Danielle, Gabrielle),
    #       - ##na (female name ending Anna, Lena, Christina)
    #       - ##ra (female name ending Laura, Sara, Barbara)
    #   - many commonly used words: as, that, some, where, what, another, possibly, otherwise, ...


    data = {
        'Names Used During Training': names_used_during_training,
        #'Names Not Used During Training': all_names_not_used_during_training,
        'Frequent Names Not Used During Training': frequent_names_not_used_during_training,
        'Unfrequent Names Not Used During Training': unfrequent_names_not_used_during_training,
        'Multiple Meanings Names Not Used During Training': multiple_meaning_names,
        'Gender Words': gender_words,
        'Other Tokens': other_tokens,
    }

    from collections import defaultdict
    df_data = defaultdict(list)
    for key, values in data.items():
        df_data['Value'] += list(values.values())
        df_data['Group'] += [f'{key} (N={len(values)})'] * len(values)

    df = pd.DataFrame(df_data)

    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(y='Group', x='Value', data=df, orient='h')
    plt.title(f'Box Plot of Token Means {model} ({part})')
    plt.xlim([0.0, 0.0001])
    plt.ylabel('Group')
    plt.xlabel('Mean Value')

    grouped = df.groupby('Group')['Value']
    stats = grouped.agg(['median', 'std']).reset_index()

    # Print statistics
    print("Statistics for each group:")
    print(stats)

    pronouns = ['himself', 'herself', 'her', 'him', 'his']
    if include_he_she:
        pronouns += ['he', 'she']

        extra_data = {
            len(data)-3: top_multiple_meaning_names.index.tolist(),
            len(data)-2: pronouns,
            len(data)-1: top_outliers.index.tolist(),
        }

        for data_offset, data in extra_data.items():
            for i, id in enumerate(sorted(data, key=lambda x: token_means[x])):
                value = token_means[id]
                plt.scatter(value, data_offset, color='r', zorder=5)  # Add a marker at the specified value on the x-axis
                offset = 0.2 if i % 2 == 0 else -0.1
                plt.annotate(id, xy=(value, data_offset), xytext=(value, data_offset + offset), textcoords='data', ha='center', rotation=0, color='r', fontsize=8)

    plt.tight_layout()
    file = f'results/img/token_means/{model}_{part}.pdf'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    plt.savefig(file, bbox_inches='tight')
    plt.show()



    from collections import defaultdict

    def cluster(weights):
        clusters = defaultdict(list)
        for name, weight in weights.items():
            if 'base_model.embeddings' in name:
                cluster = 'embedding'
            elif 'cls.predictions' in name:
                cluster = 'predictions'
            elif 'base_model.encoder.layer.' in name:
                layer = int(name.split('.')[3])
                cluster = f'layer {layer}'
            else:
                raise ValueError(f'Unknown name {name}')
            clusters[cluster].append(weight)

        return clusters


def analyze_neurons_all_parts(model, parts=None, relative=False, q=99.99, boxplot=True):
    parts = parts or ['encoder', 'decoder', 'decoder-bias']

    suffix = f'{"_boxplot" if boxplot else ""}{"_relative" if relative else ""}'

    bert_with_ae = ModelWithGradiend.from_pretrained(model)
    layer_maps = {part: {k: v.detach().cpu() for k, v in bert_with_ae.ae_named_parameters(part=part)} for part in parts}

    grads = {part: torch.concat([v.flatten() for v in layer_maps[part].values()]) for part in parts}
    grads_abs = {part: grads.abs() for part, grads in grads.items()}

    unfiltered_thresholded_layer_map = None
    if q > 0:
        threshold = {part: np.percentile(grads_abs, q=q).item() for part, grads_abs in grads_abs.items()}

        if boxplot:
            thresholded_layer_map = {
                part: {
                    k: v.abs().flatten()[v.abs().flatten() > threshold[part]]
                    for k, v in layer_map.items()
                }
                for part, layer_map in layer_maps.items()
            }
        else:
            thresholded_layer_map = {part: {k: v.abs() > threshold[part] for k, v in layer_map.items()} for part, layer_map in layer_maps.items()}
        unfiltered_thresholded_layer_map = {part: {k: v.abs() for k, v in layer_map.items()} for part, layer_map in layer_maps.items()}
    else:
        thresholded_layer_map = {parts: {k: v.abs() for k, v in layer_map.items()} for parts, layer_map in layer_maps.items()}

    all_layers = layer_maps[parts[0]].keys()
    # average for each layer
    layers = {part: {} for part in parts}
    unfiltered_layers = {part: {} for part in parts}
    for layer in reversed(all_layers):

        # retrieve layer number
        if 'layer' in layer:
            layer_number = int(layer.split('.')[3])
            key = f'Layer {layer_number}'
        elif 'cls' in layer:
            continue
        elif 'word_embeddings' in layer:
            key = 'Word Embeddings'
        elif 'position_embeddings' in layer:
            key = 'Other Embeddings'
        elif 'token_type_embeddings' in layer:
            key = 'Other Embeddings'
        # important that LayerNorm is checked after 'layer'!
        elif 'embeddings.LayerNorm' in layer:
            key = 'Other Embeddings'
        else:
            print('WARNING: Unknown layer', layer)
            continue

        for part in parts:
            if key not in layers[part]:
                layers[part][key] = []
            layers[part][key].append(thresholded_layer_map[part][layer].numpy().flatten())

            if unfiltered_thresholded_layer_map:
                if key not in unfiltered_layers[part]:
                    unfiltered_layers[part][key] = []
                unfiltered_layers[part][key].append(unfiltered_thresholded_layer_map[part][layer].numpy().flatten())


    # average for each layer
    number_of_elements = {}
    unfiltered_number_of_elements = {}
    for part in parts:
        number_of_elements[part] = {layer: sum(len(v) for v in values) for layer, values in layers[part].items()}
        unfiltered_number_of_elements[part] = {layer: sum(len(v) for v in values) for layer, values in unfiltered_layers[part].items()}

        for layer, values in layers[part].items():


            try:
                flattened = np.concat(values)
            except Exception: # todo remove (currently needed due to wrong numpy version?)
                flattened = np.array([vv for v in values for vv in v])

            if boxplot:
                layers[part][layer] = flattened
            elif relative:
                layers[part][layer] = np.mean(flattened, axis=0).item()
            else:
                layers[part][layer] = sum(flattened)

    cm = "YlGnBu"
    colors = sns.color_palette(cm, n_colors=len(parts))

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color='black')
        #plt.setp(bp['fliers'], color=color)
    def human_readable_format(num):
        if abs(num) >= 1_000_000_000:
            return f'{num / 1_000_000_000:.1f}B'
        elif abs(num) >= 1_000_000:
            return f'{num / 1_000_000:.1f}M'
        elif abs(num) >= 1_000:
            return f'{num / 1_000:.1f}k'
        else:
            return str(num)

    if boxplot:
        offset_counts = -0.4
        xlim_offset = offset_counts - 0.1
    else:
        offset_counts = 0.0
        xlim_offset = 0.0
    # Generate positions for each group on the x-axis
    # Bar width for each bar in a group
    bar_width = 0.27
    # Loop through each part and plot the bars
    for idx, (part, means) in enumerate(layers.items()):
        indices = np.arange(len(means))
        # Calculate the position of the bars for the current part
        positions = indices + idx * bar_width

        # Plot the bars
        if boxplot:
            color = colors[idx]
            flierprops = {'markersize': 4, 'color': color, 'markerfacecolor': color, 'markeredgecolor': color}
            bp = ax.boxplot(list(means.values()), positions=positions, widths=bar_width, patch_artist=True, boxprops=dict(facecolor=colors[idx], color='black'), zorder=5, vert=False, flierprops=flierprops)
            set_box_color(bp, colors[idx])

            for i, values in enumerate(means.values()):
                # Get the position of the box
                box_pos = positions[i]
                num_elements = len(values)
                # Annotate to the left of the box (adjust x-coordinate)
                ax.text(offset_counts, box_pos, f'n={human_readable_format(num_elements)}',
                        ha='left', va='center', fontsize=6, color='black')

            # add legend
            plt.plot([], label=part, color=color)
        else:
            rects = ax.barh(positions, list(means.values()), bar_width, label=part, color=colors[idx], zorder=5)


    if boxplot:
        # also report total (unfiltered number of neurons per cluster)
        xticks = [f'{k} (N={human_readable_format(unfiltered_number_of_elements[part][k])})' for k in means.keys()]
    else:
        xticks = [f'{k} (N={human_readable_format(number_of_elements[part][k])})' for k in means.keys()]
    plt.yticks(indices + bar_width, xticks)
    plt.xlim([xlim_offset, ax.get_xlim()[1]])
    plt.ylabel('Layer')
    if relative:
        plt.xlabel('Proportion of Neurons')
    elif boxplot:
        plt.xlabel('Absolute Neurons')
    elif q > 0:
        plt.xlabel('Absolute Neuron Sum')
    else:
        plt.xlabel('Relevant Neurons')
    plt.legend(loc='center right')
    plt.grid(zorder=0)
    model_name = model.split('/')[-1]
    file = f'results/img/average_neurons/{model_name}{suffix}_per_layer.pdf'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    plt.savefig(file , bbox_inches='tight')
    plt.show()

    # average for key, query, word embedding, ...

    layers = {part: {} for part in parts}
    unfiltered_layers = {part: {} for part in parts}
    for layer in all_layers:

        # retrieve layer number
        if 'embeddings' in layer:
            key = 'Embeddings'
        elif 'key' in layer:
            key = 'Key'
        elif 'query' in layer:
            key = 'Query'
        elif 'value' in layer:
            key = 'Value'
        elif 'attention.output' in layer:
            key = 'Attention Output' # attention output dense
        elif 'intermediate.dense' in layer:
            key = 'Intermediate Dense'
        elif '.output.' in layer:
            key = 'Layer Output' # layer output
        elif 'cls' in layer:
            continue
        else:
            print('WARNING: Unknown layer', layer)
            continue

        for part in parts:
            if key not in layers[part]:
                layers[part][key] = []
            layers[part][key].append(thresholded_layer_map[part][layer].numpy().flatten())

            if unfiltered_thresholded_layer_map:
                if key not in unfiltered_layers[part]:
                    unfiltered_layers[part][key] = []
                unfiltered_layers[part][key].append(unfiltered_thresholded_layer_map[part][layer].numpy().flatten())


    # average for each layer
    number_of_elements = {}
    unfiltered_number_of_elements = {}
    for part in parts:
        number_of_elements[part] = {layer: sum(len(v) for v in values) for layer, values in layers[part].items()}
        unfiltered_number_of_elements[part] = {layer: sum(len(v) for v in values) for layer, values in
                                               unfiltered_layers[part].items()}
        for layer, values in layers[part].items():
            try:
                flattened = np.concat(values)
            except Exception: # todo remove (currently needed due to wrong numpy version?)
                flattened = [vv for v in values for vv in v]

            if boxplot:
                layers[part][layer] = flattened
            elif relative:
                layers[part][layer] = np.mean(flattened, axis=0).item()
            else:
                layers[part][layer] = sum(flattened)

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop through each part and plot the bars
    for idx, (part, means) in enumerate(reversed(layers.items())):
        # reverse means
        means = dict(reversed(list(means.items())))

        indices = np.arange(len(means))
        # Calculate the position of the bars for the current part
        positions = indices + idx * bar_width

        # Plot the bars
        if boxplot:
            color = colors[idx]
            flierprops = {'markersize': 4, 'color': color, 'markerfacecolor': color, 'markeredgecolor': color}
            bp = ax.boxplot(list(means.values()), positions=positions, widths=bar_width, patch_artist=True, boxprops=dict(facecolor=colors[idx], color='black'), zorder=5, vert=False, flierprops=flierprops)
            set_box_color(bp, colors[idx])

            for i, values in enumerate(means.values()):
                # Get the position of the box
                box_pos = positions[i]
                num_elements = len(values)
                # Annotate to the left of the box (adjust x-coordinate)
                ax.text(offset_counts, box_pos, f'n={human_readable_format(num_elements)}',
                        ha='left', va='center', fontsize=8, color='black')

            # add legend
            plt.plot([], label=part, color=color)
        else:
            # Plot the bars
            ax.barh(positions, list(means.values()), bar_width, label=part, color=colors[idx], zorder=5)

    # plt.title('Average of Neurons with Absolute Gradient Above Threshold')
    if boxplot:
        # also report total (unfiltered number of neurons per cluster)
        xticks = [f'{k} (N={human_readable_format(unfiltered_number_of_elements[part][k])})' for k in means.keys()]
    else:
        xticks = [f'{k} (N={human_readable_format(number_of_elements[part][k])})' for k in means.keys()]
    plt.yticks(indices + bar_width, xticks)
    plt.xlim([xlim_offset, ax.get_xlim()[1]])
    plt.ylabel('Layer')
    if relative:
        plt.xlabel('Proportion of Neurons')
    elif boxplot:
        plt.xlabel('Absolute Neurons')
    elif q > 0:
        plt.xlabel('Absolute Neuron Sum')
    else:
        plt.xlabel('Relevant Neurons')
    plt.legend(loc='lower right')
    plt.grid(zorder=0)
    model_name = model.split('/')[-1]
    file = f'results/img/average_neurons/{model_name}{suffix}.pdf'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    plt.savefig(file, bbox_inches='tight')
    plt.show()



def analyze_models(*models, config, max_size=None, force=False, split='test', prefix=None, best_score=None, multi_task=False, ensemble=False):
    if prefix:
        # find all models in the folder with the suffix
        best_score = '_best' if best_score else ''
        models = list(models) + get_files_and_folders_with_prefix(prefix, only_folder=True, suffix=best_score)
    #print(f'Analyze {len(models)} Models:', models)

    # names_df = read_namexact(split=split)
    # df = read_genter(split=split)
    # df_no_gender = read_geneutral(max_size=10000)

    article_df = []
    for label in config['combinations']: 
        df_label = read_article_ds(split=split, article=label)
        article_df.append(df_label)

    min_len = min([len(d) for d in article_df])
    df = pd.concat(article_df)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)    

    all_dfs = []
    for gender in ['M', 'F', 'N']:
        for case in ['N', 'A', 'D', 'G']:
            label = case + gender
            df_label = read_article_ds(split=split, article=label)
            all_dfs.append(df_label)

    min_size = min([len(d) for d in all_dfs])

    all_dfs = [d.sample(n=min_size, random_state=42).reset_index(drop=True) for d in all_dfs]

    new_min_size = min(min_size, 500)
    if split == 'val':
        new_min_size = min(min_size, 50)
        print('Using smaller val size for faster eval:', new_min_size)
    print('Using balanced dataset size of:', new_min_size)
    all_dfs = [d.head(new_min_size) for d in all_dfs]


    df_all = pd.concat(all_dfs)

    neutral_data = read_de_neutral()

    if max_size:
        df = df.head(max_size)
        neutral_data = neutral_data.head(max_size)

    if split == 'val':
        neutral_data = neutral_data.head(50)


    dfs = {}
    dfs_all = {}

    for model in models:

        output = get_file_name(model, max_size=max_size, file_format='csv', split=split)
        output_all = get_file_name(model, max_size=None, inverse=True, file_format='csv', split=split, variant='all')
        if force or not (os.path.isfile(output) and os.path.isfile(output_all)):
            if ensemble: 
                ae = CombinedEncoderDecoder.from_pretrained(model)
                bert_with_ae = ModelWithGradiend.from_pretrained(model, ae=ae, ensemble=True)
            else:

                bert_with_ae = ModelWithGradiend.from_pretrained(model)
            model_analyser = DeEncoderAnalysis(config)
            analyze_df = model_analyser.analyse_encoder(bert_with_ae, df, output=output, multi_task=multi_task, neutral_data=neutral_data['text'])

            config_all = config.copy()
            config_all['articles'] = [art.lower() for art in df_all['label'].unique()]
            config_all['default_predictions'] = ['most_likely_token', 'label'] + config_all['articles']

            genders = ['M', 'F', 'N']
            cases = ['N', 'A', 'D', 'G']
            ctr = 0
            df_all_augmented_data = []
            combinations = []
            for gender in genders:
                other_genders = [g for g in genders if g != gender]
                for case in cases:
                    basic_key = case + gender
                    basic_df = df_all[df_all['dataset_label'] == basic_key]
                    other_cases = [c for c in cases if c != case]
                    label = case_gender_mapping[case][gender]
                    template_key = f'[{label.upper()}_ARTICLE]'
                    for other_gender in other_genders:
                        combination_key = f'{case}{gender}_g{other_gender}'
                        combinations.append(combination_key)
                        # todo delete if not needed and comment why this makes sense for this data

                        inverse = case_gender_mapping[case][other_gender]
                        if inverse == label:
                            continue

                        combination = {
                            'mask': template_key,
                            'inverse': inverse,
                            'code': ctr,
                            'encoding': 0, # todo?
                        }
                        with open_dict(config_all):
                            config_all[combination_key] = combination

                        g_basic_df = basic_df.copy()
                        g_basic_df['dataset_label'] = combination_key
                        g_basic_df['inverse'] = inverse
                        df_all_augmented_data.append(g_basic_df)
                        ctr += 1

                    for other_case in other_cases:
                        combination_key = f'{case}{gender}_c{other_case}'
                        combinations.append(combination_key)

                        inverse = case_gender_mapping[other_case][gender]
                        if inverse == label:
                            continue

                        combination = {
                            'mask': template_key,
                            'inverse': inverse,
                            'code': ctr,
                            'encoding': 0, # todo?
                        }
                        with open_dict(config_all):
                            config_all[combination_key] = combination
                        c_basic_df = basic_df.copy()
                        c_basic_df['dataset_label'] = combination_key
                        c_basic_df['inverse'] = inverse
                        df_all_augmented_data.append(c_basic_df)
                        ctr += 1

            config_all['combinations'] = combinations
            model_analyser_all = DeEncoderAnalysis(config_all)
            df_all_augmented = pd.concat(df_all_augmented_data).reset_index(drop=True)
            output_all = get_file_name(model, max_size=None, file_format='csv', split=split, variant='all', inverse=True)
            analyze_df_all_1 = model_analyser_all.analyse_encoder(bert_with_ae, df_all_augmented, output=output_all, multi_task=multi_task, full=True)

            ctr = 0
            df_all_augmented_data = []
            combinations = []
            for gender in genders:

                for case in cases:
                    basic_key = case + gender
                    basic_df = df_all[df_all['dataset_label'] == basic_key]

                    label = case_gender_mapping[case][gender]
                    template_key = f'[{label.upper()}_ARTICLE]'

                    combination_key = f'{case}{gender}'
                    combinations.append(combination_key)
                    # todo delete if not needed and comment why this makes sense for this data
                    inverse = label

                    combination = {
                        'mask': template_key,
                        'inverse': inverse,
                        'code': ctr,
                        'encoding': 0, # todo?
                    }
                    with open_dict(config_all):
                        config_all[combination_key] = combination

                    g_basic_df = basic_df.copy()
                    g_basic_df['dataset_label'] = combination_key
                    g_basic_df['inverse'] = inverse
                    df_all_augmented_data.append(g_basic_df)
                    ctr += 1

            config_all['combinations'] = combinations
            model_analyser_all = DeEncoderAnalysis(config_all)
            df_all_augmented = pd.concat(df_all_augmented_data).reset_index(drop=True)
            output_all = get_file_name(model, max_size=None, file_format='csv', split=split, variant='all', inverse=False)
            analyze_df_all2 = model_analyser_all.analyse_encoder(bert_with_ae, df_all_augmented, output=output_all, multi_task=multi_task, full=True)
            analyze_df_all = pd.concat([analyze_df_all_1, analyze_df_all2]).reset_index(drop=True)

            #analyze_df = analyze_model(bert_with_ae, df, names_df, output=output, df_no_gender=df_no_gender)
            print(f'Done with Model {model}')

        else:
            print(f'Skipping Model {model} as output file {output} already exists!')
            analyze_df = pd.read_csv(output)
            analyze_df_all = pd.read_csv(output_all)

        if len(models) == 1:
            return analyze_df, analyze_df_all
        dfs[model] = analyze_df
        dfs_all[model] = analyze_df_all
    return dfs, dfs_all


def highlight_highest_values(data, headers):
    # Find the highest values per column
    highest_values = {header: max([row[idx] for row in data if isinstance(row[idx], (int, float))], default=-1) for idx, header in
                      enumerate(headers) if header != 'File'}

    lowest_values = {header: min([row[idx] for row in data if isinstance(row[idx], (int, float))], default=-1) for idx, header in
                      enumerate(headers) if header != 'File'}

    # Apply ANSI escape codes to highlight the highest values
    for row in data:
        for idx, header in enumerate(headers):
            if header != 'File' and row[idx] == highest_values[header]:
                row[idx] = f'\033[1;31m{row[idx]}\033[0m'  # Bold red text
            elif header != 'File' and row[idx] == lowest_values[header]:
                row[idx] = f'\033[1;34m{row[idx]}\033[0m'  # Bold blue text

    return data


def print_all_models(folder='results/models/', *, prefix=None, keys=None, **kwargs):
    csv_files = get_files_and_folders_with_prefix(folder, suffix='.csv', only_files=True, **kwargs)
    if not csv_files:
        print('No files found!')
        return

    results = {}
    for file in csv_files:
        if prefix and prefix not in file:
            continue

        result = get_model_metrics(file)
        results[file] = result

    # if a value contains of another dict, unwrap their values by concatenating the dict keys with '_'
    for file, metrics in results.items():
        for key, value in list(metrics.items()):
            if isinstance(value, dict):
                for k, v in value.items():
                    metrics[f'{key}_{k}'] = v
                del metrics[key]

    # Prepare data for tabulate
    headers = set()
    for metrics in results.values():
        if keys is None:
            headers.update(metrics.keys())
        else:
            for key in metrics.keys():
                if key in keys:
                    headers.add(key)

    if keys:
        # sort the headers in occurrences of the header keys in keys
        ordered_keys = list(keys)
        header_keys = sorted(headers, key=lambda x: ordered_keys.index(x))
        headers = [keys[header] for header in header_keys]
    else:
        headers = sorted(headers)
        header_keys = headers

    table_data = []
    for file, metrics in sorted(results.items()):
        row = [file] + [metrics.get(header, '') for header in header_keys]
        table_data.append(row)

    headers = ['File'] + headers

    # Highlight the highest values
    table_data = highlight_highest_values(table_data, headers)

    # Print using tabulate
    print(tabulate(table_data, headers=headers, tablefmt='grid', floatfmt=".7f"))



if __name__ == '__main__':
    models = [
        'bert-base-cased',
        'bert-large-cased',
        'roberta-large',
        'distilbert-base-cased',
        'gpt2',
        'meta-llama/Llama-3.2-3B',
        'meta-llama/Llama-3.2-3B-Instruct'
    ]

    df = analyze_models(*[f'results/models/{model}' for model in models])
    print_all_models()