from tabulate import tabulate

from gradiend.evaluation.analyze_decoder import default_evaluation

from gradiend.export import models
from gradiend.util import init_matplotlib

init_matplotlib(use_tex=True)

normalizer = lambda x: x

metrics = {
    '$\mathbb{P}(F)$': lambda metrics: metrics['gender_bias_names']['avg_prob_f'],
    '$\mathbb{P}(M)$': lambda metrics: metrics['gender_bias_names']['avg_prob_m'],
    #'$\mathbb{P}(F\cup M)$': lambda metrics: metrics['gender_bias_names']['avg_prob_f'] + metrics['gender_bias_names']['avg_prob_m'],
    r'\accdec': lambda metrics: metrics['mlm']['accuracy'],
    #'APD': lambda metrics: metrics['gender_bias_names']['apd'],
    'BPI': lambda metrics: normalizer(metrics['bpi']),
    'FPI': lambda metrics: normalizer(metrics['fpi']),
    'MPI': lambda metrics: normalizer(metrics['mpi']),
}

versions = {
    'N': 'BPI',
    'F': 'FPI',
    'M': 'MPI',
}

gradiend_map = {
    'N': r'\gradiendbpi',
    'F': r'\gradiendfpi',
    'M': r'\gradiendmpi',
}

table = [[
    r'\textbf{Model}',
    r'\textbf{GF} $h$',
    r'\textbf{LR} $\alpha$',
    *[r'\textbf{' + key + r'}' for key in metrics.keys()]
    ]]


for base_model in models:
    model = f'results/models/{base_model}'

    try:
        decoder_metrics = default_evaluation(model, plot=True)
    except OSError:
        print(f'Skipping model {model} since file does not exist')
        continue

    # add base entry
    row = [models[base_model], '0.0', '0.0']
    base_metrics = decoder_metrics['base']
    for metric_name, metric_getter in metrics.items():
        row.append(f'{metric_getter(base_metrics):.3f}')
    table.append(row)

    for version_suffix, version in versions.items():
        version_stats = decoder_metrics[version.lower()]
        lr = version_stats['lr']
        feature_factor = version_stats['feature_factor']
        row = [f'\, + {gradiend_map[version_suffix]}', f'{feature_factor:.1f}', f'{lr:.0e}']
        if lr == 0 and feature_factor == 0:
            version_metrics = base_metrics
            print(f'The base model {base_model} is the best model for {version} and {base_model}')
        else:
            version_metrics = decoder_metrics[(feature_factor, lr)]

        for metric_name, metric_getter in metrics.items():
            row.append(f'{metric_getter(version_metrics):.3f}')

        table.append(row)

table = tabulate(table, headers='firstrow', tablefmt='latex_raw', disable_numparse=True)
# Split table into lines
lines = table.splitlines()

# Insert \midrule every 4th row after the header
for i in range(len(lines)-6, 3, -4):  # Start after header, then every 4th line
    lines.insert(i, r"\midrule")

# Join lines back together
table_with_midrules = "\n".join(lines)

print(table_with_midrules)