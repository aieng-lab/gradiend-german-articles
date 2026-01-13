import json
import os
import shutil
import time
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
import logging
from pathlib import Path
from pickle import dump


from gradiend.evaluation.analyze_encoder import analyze_models
from gradiend.evaluation.encoder.de_encoder_analysis import DeEncoderAnalysis
from gradiend.model import gradiend_dir
from gradiend.training.trainer import train_all_layers_gradiend, train_multiple_layers_gradiend
from gradiend.util import case_gender_mapping, RESULTS_DIR

log = logging.getLogger(__name__)

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("basename", lambda x: x.split("/")[-1])



@hydra.main(version_base=None, config_path='conf', config_name='config')
def train(cfg: DictConfig): 
    
    log.info(f"Running GRADIEND with config: {OmegaConf.to_yaml(cfg)}")

    if cfg.eval_only and cfg.model_path: 
        model_analyser = DeEncoderAnalysis(cfg.pairing)
        for split in ["val", "test"]:
            analyze_models(cfg.model_path, config=cfg.pairing, split=split, multi_task=False)
            model_analyser.get_model_metrics_m_dim(cfg.model_path, split=split)

    else:
        if cfg.mode['name'] == 'gradiend':
            train_gradiend(cfg)
        if cfg.mode['name'] == 'gradiend_ensemble':
            # this is the only one that need to construct objects first and then it can call 
            # the train function directly....
            pass






def process_label(label: str, inverse_case: str):
    assert  len(label) in {2, 3}
    assert len(inverse_case) == 1

    case = label[0]
    gender = label[1]

    article = case_gender_mapping[case][gender]

    mask = f'[{article.upper()}_ARTICLE]'
    inverse = case_gender_mapping[inverse_case][gender]

    return article, mask, inverse

def process_config(cfg):
    pairing = cfg['pairing']
    categories = pairing['categories']
    keys = list(categories.keys())

    case_chars = ['N', 'G', 'D', 'A']
    gender_chars = ['M', 'F', 'N', 'P']

    keys_all_gender = all(k in gender_chars for k in keys)
    if keys_all_gender:
        pass
    else:
        keys_all_cases = all(k in case_chars for k in keys)
        if not keys_all_cases:
            raise ValueError(f"Unexpected category keys: {keys}")

        is_multi = len(keys) > 2

        code = 0
        combinations = []
        articles = []
        for score_label, case in enumerate(keys):
            labels = categories[case]['labels']
            label_articles = []
            label_codes = []

            # temporarily unlock struct mode for modifications
            with open_dict(categories[case]):
                categories[case]['score_label'] = score_label # todo actually used????
                if is_multi:
                    encoding = [score_label]
                else:
                    encoding = 1 if score_label == 0 else -1
                categories[case]['encoding'] = encoding

            for label in labels:
                assert label.startswith(case), f"Label {label} does not match case {case}"

                if len(label) == 2:
                    other_key = keys[score_label - 1]
                else:
                    other_key = label[-1]

                article, mask, inverse = process_label(label, other_key)
                label_articles.append(article)
                label_codes.append(code)

                # unlock pairing for adding new label entries
                with open_dict(pairing):
                    pairing[label] = {
                        'mask': mask,
                        'inverse': inverse,
                        'code': code,
                        'encoding': encoding,
                    }

                code += int(not is_multi)
            code += int(is_multi)

            if is_multi:
                combinations += [f'MFN/{label[:2]}_mfn' for label in labels]
            else:
                combinations += labels
            articles += label_articles

            # add dynamically computed keys to category
            with open_dict(categories[case]):
                categories[case]['codes'] = label_codes
                categories[case]['articles'] = label_articles

        # add global keys to pairing
        with open_dict(pairing):
            pairing['combinations'] = combinations
            pairing['articles'] = list(set(articles))

    if cfg.get('augment_neutral', False):
        all_combinations = ['NM', 'NF', 'NN', 'GM', 'GF', 'GN', 'DM', 'DF', 'DN', 'AM', 'AF', 'AN']
        unused_combinations = [comb for comb in all_combinations if comb not in pairing['combinations']]
        if unused_combinations:
            print(f'Adding neutral combinations: {unused_combinations}')
            neutral_category = {
                'labels': unused_combinations,
                'articles': ['der', 'die', 'das', 'dem', 'des', 'den'],
                'codes': list(range(len(pairing['articles']), len(pairing['articles']) + len(unused_combinations))),
                'encoding': 0,
                'score_label': 2,
            }
            with open_dict(pairing['categories']):
                pairing['categories']['Neutral'] = neutral_category
            with open_dict(pairing):
                pairing['plot_name'] += '_neutral_augmented'

            pairing['combinations'] += unused_combinations
            pairing['articles'] = neutral_category['articles'] # all articles anyways

            for i, comb in enumerate(unused_combinations):
                article, mask, inverse = process_label(comb, comb[0])
                with open_dict(pairing):
                    pairing[comb] = {
                        'mask': mask,
                        'inverse': article, # we dont use inverse for neutral!
                        'code': i + len(pairing['articles']),
                        'encoding': 0,
                    }


    token_to_ignore = list(set(pairing.get('token_to_ignore', []) + pairing['articles']))
    token_to_ignore += [token.capitalize() for token in pairing['articles'] if token[0].islower()]
    token_to_ignore += [token.upper() for token in pairing['articles'] if token.isalpha()]
    token_to_ignore = list(set(token_to_ignore))

    with open_dict(pairing):
        pairing['token_to_ignore'] = token_to_ignore
        pairing['default_predictions'] = pairing['articles'] + ['most_likely_token', 'label']

    return cfg


def train_gradiend(cfg: DictConfig):

    cfg = process_config(cfg)


    base_path = Path.cwd()
    version = cfg.get('version', None)
    experiments_path = base_path / RESULTS_DIR / "experiments" / cfg.mode['name']
    metric = cfg.mode['metric']

    metrics = []
    total_start = time.time()
    times = []

    model_analyser = DeEncoderAnalysis(cfg.pairing)

    if version is None or version == '':
        version = ''
    else:
        version = f'/v{version}'

    # if torch.cuda.is_available():
    #     log.info("CUDA is available, starting memory snapshot...")

    # torch.cuda.memory._record_memory_history(enabled='all')

    analyze_after_all_trainings = True

    base_model_id = cfg.base_model.split('/')[-1]
    base_output = experiments_path / cfg.pairing.plot_name / f"dim_{cfg.mode.model_config.num_dims}_{cfg.mode.model_config.source}" / base_model_id

    num_runs = cfg.num_runs

    for i in range(num_runs, 4):
        folder = base_output / f"{i}"
        if os.path.exists(folder):
            num_runs += 1
            print('Increasing num_runs to', num_runs, 'as', folder, 'exists')
        else:
            break

    for i in range(cfg.num_runs):
        log.info(f"Run {i} for GRADIEND")
        start = time.time()
        output = base_output / f"{i}"
        metrics_file = f'{output}/metrics.json'
        base_model = cfg.base_model
        if os.path.exists(metrics_file):
            metrics.append(abs(json.load(open(metrics_file))))
            print(f'Skipping training of {output} as it already exists')
        else:
            if not os.path.exists(output):
                print('Training', output)
                run_config = cfg.mode.model_config.copy()
                offset_seed = run_config.get('seed', 0)
                run_config.seed = i + offset_seed

                if 'layers' in run_config:
                    train_multiple_layers_gradiend(model=base_model, output=output, **run_config)
                else:
                    train_all_layers_gradiend(config=cfg.pairing, model=base_model, output=output, **run_config)
            else:
                print('Model', output, 'already exists, skipping training, but evaluate')

        model_bin = os.path.join(output, 'pytorch_model.bin')
        if os.path.exists(model_bin) and not analyze_after_all_trainings:
            log.info(f"Evaluating models in {output} for test set")
            analyze_models(str(output), config=cfg.pairing, split='test', multi_task=False)

        if cfg.mode.model_config.num_dims > 1:
            if not analyze_after_all_trainings:
                #model_analyser.get_model_metrics_m_dim(output, split='val')
                model_analyser.get_model_metrics_m_dim(output, split='test')
        else:
            if not analyze_after_all_trainings:
                model_analyser.get_model_metrics(output, split='test')
            if not os.path.isfile(metrics_file):
                print('Evaluating metrics for', output) # todo remove, but this is currently needed if metrics.json was not saved after training
                log.info(f"Analyzing models in {output} for validation set")
                analyze_models(str(output), config=cfg.pairing, split='val', multi_task=False)
                model_metrics = model_analyser.get_model_metrics(output, split='val')
                metric_value = abs(model_metrics[metric])
                json.dump(metric_value, open(metrics_file, 'w'))
                metrics.append(metric_value)


        print(f'Metrics for model {base_model}: {metrics}')
    best_index = np.argmax(metrics)
    print('Best metric at index', best_index, 'with value', metrics[best_index])

    best_dir = gradiend_dir(base_output)
    base_dir = os.path.abspath(best_dir)

    # other dirs to clean up (containing at least one file other than metrics.json)
    other_dirs = [
        os.path.join(base_output, d)
        for d in os.listdir(base_output)
        if os.path.isdir(os.path.join(base_output, d))
           and os.path.abspath(os.path.join(base_output, d)) != base_dir
           and len(os.listdir(os.path.join(base_output, d))) > 1
    ]

    print(f'Keeping best model directory: {base_dir}')
    if other_dirs:
        print(f'Cleaning up other directories: {other_dirs}')
        for d in other_dirs:
            d = Path(d)
            for p in d.iterdir():
                if p.name == "metrics.json":
                    continue
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)

    if analyze_after_all_trainings:
        log.info(f"Evaluating models in {best_dir} for test set")
        analyze_models(best_dir, config=cfg.pairing, split='test', multi_task=False)

        if cfg.mode.model_config.num_dims > 1:
            #model_analyser.get_model_metrics_m_dim(base_output, split='val')
            model_analyser.get_model_metrics_m_dim(best_dir, split='test')
        else:
            #model_metrics = model_analyser.get_model_metrics(base_output, split='val')
            model_metrics = model_analyser.get_model_metrics(best_dir, split='test')
            #metric_value = model_metrics[metric]
            #json.dump(metric_value, open(metrics_file, 'w'))
            #metrics.append(metric_value)

    return output


if __name__ == '__main__':
    train()