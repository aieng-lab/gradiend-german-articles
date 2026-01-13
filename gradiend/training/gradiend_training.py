import json
import os
import shutil
import time

import numpy as np
import torch

from gradiend.evaluation.analyze_encoder import analyze_models
from gradiend.evaluation.select_models import select
from gradiend.training import train_all_layers_gradiend, train_multiple_layers_gradiend, PolarFeatureLoss


def train(base_model,config, model_config, dim, n=3, metric='pearson', multi_task=False, force=True, version=None, clear_cache=False):
    det_combination = config['plot_name']
    lr =model_config['lr']
    epochs = model_config['epochs']
    # metric = f"{metric}_{det_combination}"
    metrics = []
    total_start = time.time()
    times = []
 
    if version is None or version == '':
        version = ''
    else:
        version = f'/v{version}'

    for i in range(n):
        start = time.time()
        output = f'results/experiments/gradiend/{det_combination}/acc_only/dim_{dim}/{lr}/{epochs}/{base_model}/{version}/{i}'
        metrics_file = f'{output}/metrics.json'
        if not force and os.path.exists(metrics_file):
            metrics.append(json.load(open(metrics_file)))
            print(f'Skipping training of {output} as it already exists')
            continue

        if not os.path.exists(output):
            print('Training', output)
            model_config['seed'] = i
            if 'layers' in model_config:
                train_multiple_layers_gradiend(model=base_model, output=output, **model_config)
            else:
                train_all_layers_gradiend(config=config, multi_task=multi_task, model=base_model, output=output, **model_config)
        else:
            print('Model', output, 'already exists, skipping training, but evaluate')

        #TODO maybe create a strategy pattern-like context with analyze_models -> picks the 'Strategy' 
        analyze_models(output, config=config, split='val', force=force, multi_task=multi_task)
        #model_metrics = model_analyser.get_model_metrics(output, split='val')
        #metric_value = model_metrics[metric]
        #json.dump(metric_value, open(metrics_file, 'w'))
        #metrics.append(metric_value)

        times.append(time.time() - start)

        if clear_cache:
            cache_folder = f'data/cache/gradients/{det_combination}/{base_model}'
            if os.path.exists(cache_folder):
                shutil.rmtree(cache_folder)

    #print(f'Metrics for model {base_model}: {metrics}')
    # best_index = np.argmax(metrics)
    # print('Best metric at index', best_index, 'with value', metrics[best_index])

    base_model_id = base_model.split('/')[-1]
    output = f'results/models/{det_combination}/acc_only/dim_{dim}/{lr}/{epochs}/{base_model_id}{version.replace("/", "-")}'
    # copy the best model to output
    shutil.copytree(f'results/experiments/gradiend/{det_combination}/acc_only/dim_{dim}/{lr}/{epochs}/{base_model}{version}', output, dirs_exist_ok=True)

    total_time = time.time() - total_start
    if times:
        print(f'Trained {len(times)} models in {total_time}s')
        print(f'Average time per model: {np.mean(times)}')
    else:
        print('All models were already trained before!')

    return output


if __name__ == '__main__':
    model_configs = {
        'bert-base-cased': dict(),
        'bert-large-cased': dict(eval_max_size=0.5, eval_batch_size=4),
        'distilbert-base-cased': dict(),
        'roberta-large': dict(eval_max_size=0.5, eval_batch_size=4),
        'gpt2': dict(),
        'meta-llama/Llama-3.2-3B-Instruct': dict(batch_size=32, eval_max_size=0.05, eval_batch_size=1, epochs=1, torch_dtype=torch.bfloat16, lr=1e-4),
        'meta-llama/Llama-3.2-3B': dict(batch_size=32, eval_max_size=0.05, eval_batch_size=1, epochs=1, torch_dtype=torch.bfloat16, lr=1e-4, n_evaluation=250),
    }

    models = []
    for base_model, model_config in model_configs.items():
        model = train(base_model, model_config, n=3, version='', clear_cache=False, force=False)
        models.append(model)

    for model in models:
        select(model)
