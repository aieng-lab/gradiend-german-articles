import gc
import random
import shutil

from matplotlib.lines import Line2D
import numpy as np
from matplotlib import pyplot as plt

import time
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

import logging
from gradiend.model import ModelWithGradiend

import datetime
import os

from gradiend.training.de_training_dataset import create_de_training_dataset, create_de_eval_dataset_no_cache, OversampledSingleLabelBatchSampler
from gradiend.util import hash_it

log = logging.getLogger(__name__)


# Create a unique directory for each run based on the current time
def get_log_dir(base_dir="logs", output=''):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_dir, output + f'_{current_time}')
    return log_dir

def invert(category_to_invert, other_category, category_means):
    """
    Determine whether to invert encoding based on category means.

    Args:
        category_to_invert (str): The category to check for inversion.
        category_means (dict): Mapping of category -> mean value.

    Returns:
        bool: True if the category should be inverted, False otherwise.
    """
    mean_to_invert = category_means[category_to_invert]

    mean_other = category_means[other_category]

    return mean_to_invert < -0.1 and mean_other > 0.1


# Define the custom loss function
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha

    def forward(self, outputs, targets):
        mse = self.mse_loss(outputs, targets)
        l1 = self.l1_loss(outputs, targets)
        return mse + self.alpha * l1

    def __str__(self):
        return f'CombinedLoss(alpha={self.alpha})'

class PolarFeatureLoss(nn.Module):
    def __init__(self, alpha=0.001):
        super(PolarFeatureLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, output, target, encoded_value):
        mse_loss = self.mse(output, target)
        reg_term = 1.0 - torch.abs(encoded_value)
        loss = mse_loss + self.alpha * reg_term * mse_loss
        print(f'MSE Loss: {mse_loss.item()}, Regularization Term: {reg_term.item()}, Encoded Value: {encoded_value.item()}')
        return loss


"""
Function: train

Trains a BERT-based autoencoder model using gradient-based input representations with evaluation, and logging functionalities.

Parameters:

model_with_gradiend: ModelWithGradiend instance of the model to train
output (str, default='results/models/gradiend'): Path to save trained model.
checkpoints (bool, default=False): Enables intermediate checkpointing every 5000 steps.
max_iterations (int, optional): Maximum number of training iterations.
criterion_ae (nn.Module, default=nn.MSELoss()): Loss function for autoencoder.
batch_size (int, default=32): Training batch size.
batch_size_data (bool or int, default=True): If the training data is batched, i.e., only a single gender is used for the training. If True, uses batch_size for data loading.
source (str, default='gradient'): Type of input data ('gradient', 'inv_gradient', 'diff').
target (str, default='diff'): Type of target data ('gradient', 'inv_gradient', 'diff').
epochs (int, default=1): Number of training epochs.
neutral_data (bool, default=False): Uses gender-neutral data if True. This gender-neutral data is then also used for the training.
neutral_data_prop (float, default=0.5): Proportion of neutral data when neutral_data=True.
plot (bool, default=False): Enables visualization of autoencoder training.
n_evaluation (int, default=250): Evaluation frequency in training steps. The evaluation is based on the GRADIEND encoder.
lr (float, default=1e-5): Learning rate.
weight_decay (float, default=1e-2): Weight decay for optimizer.
do_eval (bool, default=True): Enables evaluation during training.
keep_only_best (bool, default=True): Retains only the best-performing model.
eval_max_size (int, optional): Maximum size of evaluation datasets.
eval_batch_size (int, default=32): Batch size for evaluation.
eps (float, default=1e-8): Epsilon for numerical stability in optimization.
normalized (bool, default=True): Normalizes encoded values if True.

Saves best model in output_best directory.

Returns:

output (str): Path where the trained model is saved.
"""
def train(model_with_gradiend,
          config,
          multi_task= False,
          shared = False,
          num_dims = 1,
          accumulation_steps = 1,
          output='results/models/gradiend',
          checkpoints=False,
          max_iterations=None,
          criterion_ae=nn.MSELoss(),
          batch_size=32,
          batch_size_data=True,
          source='inv_gradient',
          target='diff',
          epochs=1,
          neutral_data=False,
          neutral_data_prop=0.5,
          plot=False,
          n_evaluation=250,
          lr=1e-5,
          weight_decay=1e-2,
          do_eval=True,
          keep_only_best=True,
          eval_max_size=None,
          eval_batch_size=4,
          eps=1e-8,
          normalized=True,
          use_cached_gradients=False,
          torch_dtype=torch.float32,
          ensemble=False,
          ):

    print('Training GRADIEND model')
    print('Output:', output)
    print('Batch size:', batch_size)
    print('Learning rate:', lr)
    print('Source', source)
    print('Dtype:', torch_dtype)

    log.info(f"Training GRADIEND")

    if model_with_gradiend.base_model.dtype != torch_dtype:
        model_with_gradiend = model_with_gradiend.to(dtype=torch_dtype)

    # Load pre-trained BERT model and tokenizer
    tokenizer = model_with_gradiend.tokenizer
    is_generative = model_with_gradiend.is_generative

    # Create a datasets and dataloader for BERT inputs

    if batch_size_data is True:
        batch_size_data = batch_size

    train_datasets = []
    for label in config['combinations']: 
        train_dataset = create_de_training_dataset(
        tokenizer, config=config, max_size=None, split='train', article=label, is_generative=False, multi_task=multi_task)
        train_datasets.append(train_dataset)    


    dataset = torch.utils.data.ConcatDataset(train_datasets)  
    dataloader = DataLoader(dataset, batch_sampler=OversampledSingleLabelBatchSampler(dataset, batch_size=batch_size))
    #dataloader = DataLoader(dataset, batch_sampler=SingleLabelBatchSampler(dataset, batch_size=batch_size))

    
    if do_eval:
        eval_data = create_de_eval_dataset_no_cache(
            model_with_gradiend, config=config, split='val', source=source, eval_max_size=eval_max_size, is_generative=is_generative, multi_task=multi_task)

        def _evaluate(data):
            start = time.time()

            device = model_with_gradiend.gradiend.device_encoder
            encoded = []
            if isinstance(data, dict):
                with torch.no_grad():
                    # even if other source is chosen, the correct gradients are loaded into eval_data
                    gradients = data['gradients']
                    labels = data['labels'].values()
                    encodings = data['encodings']
                    grads = list(gradients.values())

                    if eval_batch_size > 1:
                        for i in range(0, len(grads), eval_batch_size):
                            batch = grads[i:min(i + eval_batch_size, len(grads))]
                            batch_on_device = [g.to(device, dtype=torch_dtype) for g in batch]

                            encoded_values = model_with_gradiend.gradiend.encoder(torch.stack(batch_on_device))

                            encoded.extend(encoded_values.detach().to(torch.float32).cpu().numpy().tolist())

                            # free memory on device
                            del batch_on_device
                            torch.cuda.empty_cache()  # if using GPU, it helps clear memory
                    else:
                        for grads in gradients.values():
                            encoded_value = model_with_gradiend.gradiend.encoder(grads.to(device, dtype=torch_dtype))
                            encoded.append(encoded_value.detach().to(torch.float32).cpu().numpy().tolist())
            else:
                if eval_batch_size > 1:
                    print(f"WARNING: eval batch size > 1 (is {eval_batch_size}) currently not supported ")
                labels = []
                encodings = {}
                for entry in data:
                    grads = entry['grad']
                    labels.append(entry['label'])
                    encodings[entry['text']] = entry['encoding']
                    with torch.no_grad():
                        encoded_value = model_with_gradiend.gradiend.encoder(grads.to(device, dtype=torch_dtype))
                        encoded.append(encoded_value.detach().to(torch.float32).cpu().numpy().tolist())
           
            #TODO move with config
            #score_labels = {k: 1 if v in [0, 1, 2, 3] else 0 if v in [4,5,6,7] else 2 for k, v in labels.items()}
            score_labels = encodings.values()
            if num_dims > 1:
                if ensemble:
                    encoded = np.array(encoded).mean(axis=1)
                else:
                    encoded = [np.mean(e)[0] for e in encoded]
            else:
                encoded = [e[0] for e in encoded]

            score = -pearsonr(list(score_labels), encoded).correlation

            # also compute score for encoded != 0
            labels_unequal_0_indices = [i for i, e in enumerate(score_labels) if e != 0.0]
            encoded_unequal_0 = [e for i, e in enumerate(encoded) if i in labels_unequal_0_indices]
            score_labels_unequal_0 = [label for i, label in enumerate(score_labels) if i in labels_unequal_0_indices]
            score_unequal_0 = -pearsonr(list(score_labels_unequal_0), encoded_unequal_0).correlation


            gender_keys = list(config['categories'].keys())
            category_to_invert = max(config['categories'], key=lambda cat: config['categories'][cat]['encoding'])
            other_category = min(config['categories'], key=lambda cat: config['categories'][cat]['encoding'])
            if category_to_invert == other_category:
                raise ValueError('Category to invert and other category are the same!', category_to_invert)

            log.info(f"Category to invert: {category_to_invert}")
            category_means = {}

            for gender_key in gender_keys:
                codes = config['categories'][gender_key]['codes']
                label_means = [np.mean([e for e, label in zip(encoded, labels) if label == code]) for code in codes]
                category_means[gender_key] = np.mean(label_means)


            #TODO this (what should be negative and positive) is currently decided on which article comes first in the config
            if normalized and invert(category_to_invert, other_category, category_means) and num_dims == 1:
                log.info(f"Invert encoding since {category_to_invert} encoded value is {category_means[category_to_invert]:.6f}")
                model_with_gradiend.invert_encoding()
                score = -score
                score_unequal_0 = -score_unequal_0
                for gender_key in gender_keys:
                    category_means[gender_key] = - category_means[gender_key]


            if np.isnan(score):
                score = 0.0
                score_unequal_0 = 0.0

            end = time.time()

            log.info(f"Encoded means: {category_means}")
            print(f'Evaluated in {(end - start):.2f}s')
            for gender_key, mean_value in category_means.items():
                print(f"Mean encoded value for {gender_key}: {mean_value:6f}")


            return score, score_unequal_0, *tuple(category_means.values())



        def evaluate():
            score_ = _evaluate(eval_data)
            return score_
    else:
        if normalized:
            raise ValueError('Normalization is only possible if evaluation is enabled!')

        # dummy evaluation function
        def evaluate():
            return None

    is_llama = 'llama' in model_with_gradiend.base_model.name_or_path.lower()


    # Training loop
    start_time = time.time()
    global_step = 0
    last_losses = []
    last_losses2 = []
    scores = []
    scores_without_0 = []
    losses = []
    losses2 = []
    encoder_changes = []
    decoder_changes = []
    encoder_norms = []
    decoder_norms = []
    mean_males = []
    mean_females = []
    max_losses = 100
    max_losses2 = 1000 # keep track of a 2nd moving average for compatibility reasons
    convergence = None
    best_score_checkpoint = None
    score = 0.0
    score_without_0 = 0.0
    total_training_time_start = time.time()
    training_data_prep_time = 0.0
    training_gradiend_time = 0.0
    eval_time = 0.0
    cache_dir = ''
    if use_cached_gradients:
        model_id = os.path.basename(model_with_gradiend.base_model.name_or_path)
        layers_hash = model_with_gradiend.layers_hash()
        cache_dir = f'results/cache/training/gradiend/{model_id}/{layers_hash}'

    optimizer_ae = torch.optim.AdamW(model_with_gradiend.gradiend.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    #optimizer_ae = PCGrad(torch.optim.AdamW(model_with_gradiend.gradiend.parameters(), lr=lr, weight_decay=weight_decay, eps=eps))

    if plot:
        fig = plt.figure(figsize=(12, 6))
        model_with_gradiend.gradiend.plot(fig=fig)
        plt.pause(0.1)
    else:
        fig = None

    len_dataloader = len(dataloader)
    for epoch in range(epochs):

        if max_iterations and global_step >= max_iterations:
            convergence = f'max iterations ({max_iterations}) reached'
            print(f'Stopping training since max iterations ({max_iterations}) reached')
            break

        dataloader_iterator = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
         

        for i, batch in enumerate(dataloader_iterator):

            ####### Data Preparation ########
            data_prep_start = time.time()
            cache_file = ''
            if use_cached_gradients:
                hash = hash_it(batch['text'])
                cache_file = f'{cache_dir}/{hash}.pt'

            if use_cached_gradients and os.path.exists(cache_file):
                factual_gradients, counterfactual_gradients = torch.load(cache_file)
                # todo actually save gradients!

            else:

                factual_inputs = batch[True]
                counterfactual_inputs = batch[False]
                inverse = batch['inverse']
                determiner = batch['inverse']


                factual_gradients = None
                counterfactual_gradients = None

                gradients_keywords = {'gradient', 'diff'}
                inv_gradients_keywords = {'inv_gradient', 'diff'}

                source = source.strip()
                target = target.strip()

                if source in gradients_keywords or target in gradients_keywords:
                    factual_gradients = model_with_gradiend.forward_pass(factual_inputs)

                if source in inv_gradients_keywords or target in inv_gradients_keywords:
                    if multi_task:
                        multi_task_counterfactual_gradients = []
                        for counterfactual_input in counterfactual_inputs:
                            grad_output = model_with_gradiend.forward_pass(counterfactual_input)
                            multi_task_counterfactual_gradients.append(grad_output)
                    else:
                        counterfactual_gradients = model_with_gradiend.forward_pass(counterfactual_inputs)

                del factual_inputs
                del counterfactual_inputs

            if source == 'gradient':
                source_tensor = factual_gradients
            elif source == 'diff':
                source_tensor = factual_gradients - counterfactual_gradients
            elif source == 'inv_gradient':
                if multi_task: 
                    source_tensor = multi_task_counterfactual_gradients
                else:
                    source_tensor = counterfactual_gradients
            else:
                raise ValueError(f'Unknown source: {source}')

            target_tensor = factual_gradients
            if target == 'inv_gradient':
                target_tensor = counterfactual_gradients
            elif target == 'diff':
                # if config neutral augmented and this is neutral augmented batch, actually use factual


                if multi_task:
                    target_tensors = []
                    for counterfactual_gradient in multi_task_counterfactual_gradients:
                        grad_target = target_tensor - counterfactual_gradient
                        target_tensors.append(grad_target)
                else:        
                    target_tensor -= counterfactual_gradients
            elif target == 'gradient':
                break # target_tensor is already set
            else:
                raise ValueError(f'Unknown target: {target}')
            training_data_prep_time += time.time() - data_prep_start


            del factual_gradients
            del counterfactual_gradients
            if multi_task:
                del multi_task_counterfactual_gradients

            ######## Gradiend Training ########
            gradiend_start = time.time()

            # Forward pass through autoencoder
            # if multi_task: 
            #     source_tensor= [
            #         t if t.device == model_with_gradiend.gradiend.device_encoder else t.to(model_with_gradiend.gradiend.device_encoder)
            #         for t in source_tensor]
            # else:
            if source_tensor.device != model_with_gradiend.gradiend.device_encoder:
                source_tensor = source_tensor.to(model_with_gradiend.gradiend.device_encoder)
            
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                if multi_task and source == 'inv_gradient': 
                    all_outputs = []
                    all_encoded_values = []
                    for tensor in source_tensor: 
                        outputs_ae, encoded_value = model_with_gradiend.gradiend(tensor, return_encoded=True)
                        all_outputs.append(outputs_ae)
                        all_encoded_values.append(encoded_value)
                else:
                    outputs_ae, encoded_value = model_with_gradiend.gradiend(source_tensor, return_encoded=True)

                del source_tensor

                # calculate loss
                if target_tensor.device != outputs_ae.device:
                    target_tensor = target_tensor.to(outputs_ae.device)

                if multi_task:
                    target_tensors = [
                        t if t.device == outputs_ae.device else t.to(outputs_ae.device)
                        for t in target_tensors 
                        ]
                    # target_tensors = [t.to(outputs_ae.device) for t in target_tensors]

                # release memory
                torch.cuda.empty_cache()
                gc.collect()


                if isinstance(criterion_ae, PolarFeatureLoss):
                    loss_ae = criterion_ae(outputs_ae, target_tensor, encoded_value)
                else:
                    if multi_task:
                        if source == 'gradient': 
                            task_losses = []
                            for grad_target in target_tensors: 
                                task_loss_ae = criterion_ae(outputs_ae, grad_target)
                                task_losses.append(task_loss_ae)

                            loss_ae = torch.stack(task_losses).mean()
                            # optimizer_ae.pc_backward(task_losses)
                        elif source == 'inv_gradient': 
                            task_losses = []
                            for grad_output_ae, grad_target in zip(all_outputs, target_tensors):
                                task_loss_ae = criterion_ae(grad_output_ae, grad_target)
                                task_losses.append(task_loss_ae)

                            loss_ae = torch.stack(task_losses).mean() 
                            # optimizer_ae.pc_backward(task_losses)
                    else:     
                        loss_ae = criterion_ae(outputs_ae, target_tensor)
                del target_tensor
                

                loss_ae = loss_ae / accumulation_steps
                if is_llama:
                    del batch
                    del encoded_value
                    # free memory
                    torch.cuda.empty_cache()
                    gc.collect()

                if loss_ae is not None:
                    # this could be the case, e.g., if the mask is outside of the input (should rarely happend)
                    loss_ae.backward()

                if is_llama:
                    del outputs_ae
                    torch.cuda.empty_cache()
                    gc.collect()

        
                if (i + 1) % accumulation_steps == 0  or (i + 1 == len(dataloader)):
                    optimizer_ae.step()
                    optimizer_ae.zero_grad()
                    
                    # gc.collect()  # Run Python garbage collection
                    # torch.cuda.empty_cache()  # Release unused memory back to CUDA driver
            
            loss_ae = loss_ae.item()
            training_gradiend_time += time.time() - gradiend_start

       
            if len(last_losses) < max_losses:
                last_losses.append(loss_ae)
            else:
                last_losses = last_losses[1:] + [loss_ae]

            if len(last_losses2) < max_losses2:
                last_losses2.append(loss_ae)
            else:
                last_losses2 = last_losses2[1:] + [loss_ae]

            if max_iterations and global_step >= max_iterations:
                convergence = f'max iterations ({max_iterations}) reached'
                break

            global_step += 1
            last_iteration = global_step == max_iterations or (epoch == epochs - 1 and i == len_dataloader - 1)
            if do_eval and ((i+1) % n_evaluation == 0 or i == 0)or last_iteration:
                eval_start = time.time()
                score, score_without_0, mean_male, mean_female, *rest = evaluate()
                scores.append(score)
                scores_without_0.append(score_without_0)
                mean_males.append(mean_male)
                mean_females.append(mean_female)
                eval_time += time.time() - eval_start

            n_loss_report = n_evaluation if n_evaluation > 0 else 100
            if ((i+1) % n_loss_report == 0 or  i == 0) or last_iteration:
                # validate on small validation set
                mean_loss = sum(last_losses) / len(last_losses)
                mean_loss2 = sum(last_losses2) / len(last_losses2)
                encoder_norm = model_with_gradiend.gradiend.encoder_norm
                decoder_norm = model_with_gradiend.gradiend.decoder_norm
                avg_grad_norm = model_with_gradiend.gradiend.avg_gradient_norm
                output_str = f'Epoch [{epoch + 1}/{epochs}], Loss AE: {mean_loss:.10f}, Correlation score: {score:.6f} (without 0 {score_without_0:.6f}), encoder {encoder_norm}, decoder {decoder_norm}, avg grad norm {avg_grad_norm}'
                if hasattr(model_with_gradiend.gradiend, 'encoder_change'):
                    encoder_change = model_with_gradiend.gradiend.encoder_change
                    decoder_change = model_with_gradiend.gradiend.decoder_change
                    output_str += f'encoder change {encoder_change}, decoder change {decoder_change}'
                    encoder_changes.append(encoder_change)
                    decoder_changes.append(decoder_change)
                
                log.info(output_str)
                log.info(f'Rest of evaluate {rest} ')
                print(output_str)
                losses.append(mean_loss)
                losses2.append(mean_loss2)
                encoder_norms.append(encoder_norm)
                decoder_norms.append(decoder_norm)

                if best_score_checkpoint is None or abs(score) >= abs(best_score_checkpoint['score']):
                    if best_score_checkpoint is None:
                        print('First score:', score, 'at global step', global_step)
                    elif abs(score) > abs(best_score_checkpoint['score']):
                        print('New best score:', score, 'at global step', global_step)
                    else:
                        print('Same score:', score, 'at global step', global_step)
                    best_score_checkpoint = {
                        'score': score,
                        'global_step': global_step,
                        'epoch': epoch,
                        'loss': mean_loss
                    }
                    # save checkpoint
                    training_information = {
                        'max_iterations': max_iterations,
                        'convergence': convergence,
                        'batch_size': batch_size,
                        'accumulation_steps': accumulation_steps,
                        'batch_size_data': batch_size_data,
                        'criterion_ae': str(criterion_ae),
                        'activation': str(model_with_gradiend.gradiend.activation),
                        'output': str(output),
                        'base_model': model_with_gradiend.base_model.name_or_path,
                        'layers': model_with_gradiend.gradiend.layers,
                        'score': score,
                        'scores': scores,
                        'scores_without_0': scores_without_0,
                        'mean_males': mean_males,
                        'mean_females': mean_females,
                        'losses': losses,
                        'losses_1000': losses2,
                        'encoder_changes': encoder_changes,
                        'decoder_changes': decoder_changes,
                        'encoder_norms': encoder_norms,
                        'decoder_norms': decoder_norms,
                        'time': time.time() - start_time,
                        'best_score_checkpoint': best_score_checkpoint,
                        'bias_decoder': model_with_gradiend.gradiend.bias_decoder,
                        'epoch': epoch,
                        'n_evaluation': n_evaluation,
                        'lr': lr,
                        'weight_decay': weight_decay,
                        'source': source,
                        'target': target,
                        'global_step':  global_step,
                        'eval_max_size': eval_max_size,
                        'eval_batch_size': eval_batch_size,
                        'eps': eps,
                        'combinations': list(config['combinations']),
                        'training_data_prep_time': training_data_prep_time,
                        'training_gradiend_time': training_gradiend_time,
                        'eval_time': eval_time,
                        'total_training_time': time.time() - total_training_time_start,
                    }

                    model_with_gradiend.save_pretrained(f'{str(output)}_best', training=training_information)

            if i > 0:
                if plot and i % 1000 == 0:
                    model_with_gradiend.gradiend.plot(fig=fig, n=i)
                    plt.pause(0.1)


                if checkpoints and global_step % 5000 == 0:
                    model_name = f'{output}_{global_step}'
                    model_with_gradiend.save_pretrained(model_name, convergence=convergence)
                    print('Saved intermediate result')

        training_information = {
            'max_iterations': max_iterations,
            'convergence': convergence,
            'batch_size': batch_size,
            'batch_size_data': batch_size_data,
            'criterion_ae': str(criterion_ae),
            'activation': str(model_with_gradiend.gradiend.activation),
            'output': str(output),
            'base_model': model_with_gradiend.base_model.name_or_path,
            'layers': model_with_gradiend.gradiend.layers,
            'score': score,
            'scores': scores,
            'mean_males': mean_males,
            'mean_females': mean_females,
            'losses': losses,
            'losses_1000': losses2,
            'encoder_changes': encoder_changes,
            'decoder_changes': decoder_changes,
            'encoder_norms': encoder_norms,
            'decoder_norms': decoder_norms,
            'time': time.time() - start_time,
            'best_score_checkpoint': best_score_checkpoint,
            'bias_decoder': model_with_gradiend.gradiend.bias_decoder,
            'epoch': epoch,
            'n_evaluation': n_evaluation,
            'lr': lr,
            'weight_decay': weight_decay,
            'source': source,
            'target': target,
            'global_step': global_step,
            'eval_max_size': eval_max_size,
            'eval_batch_size': eval_batch_size,
            'eps': eps,
            'combinations': list(config['combinations']),
            'training_data_prep_time': training_data_prep_time,
            'training_gradiend_time': training_gradiend_time,
            'eval_time': eval_time,
            'total_training_time': time.time() - total_training_time_start,
        }

        model_with_gradiend.gradiend.save_pretrained(output, training=training_information)
        print('Saved the auto encoder model as', output)
        print('Best score:', best_score_checkpoint)

        try:
            import humanize
            import datetime

            def humanize_time(seconds):
                return humanize.naturaldelta(datetime.timedelta(seconds=seconds))
            print(f'Epoch {epoch + 1}/{epochs} finished')
            print('Total Training time:', humanize_time(training_information['total_training_time']))
            print('Training data preparation time:', humanize_time(training_information['training_data_prep_time']))
            print('Training Evaluation time:', humanize_time(training_information['eval_time']))
            print('Training GRADIEND time:', humanize_time(training_information['training_gradiend_time']))
        except ModuleNotFoundError:
            print('Please install humanize to get a human-readable training time')

    if plot:
        plt.show()

    log.info('Best score:', best_score_checkpoint)
    print('Best score:', best_score_checkpoint)

    # release memory
    del model_with_gradiend

    # Call garbage collector
    gc.collect()

    # Empty the CUDA cache
    torch.cuda.empty_cache()

    if keep_only_best:
        # delete the output folder
        shutil.rmtree(output)

        # rename the output_best folder to output
        os.rename(f'{output}_best', output)


    # save metrics.json containing metric value in output
    metrics_file = os.path.join(output, 'metrics_train.json')
    import json
    with open(metrics_file, 'w') as f:
        json.dump(score, f)

    print('Saved the auto encoder model as', output)
    return output

def create_bert_with_ae(model, layers=None, activation='tanh', activation_decoder=None, bias_decoder=True, grad_iterations=1, decoder_factor=1.0, seed=0, torch_dtype=torch.float32, num_dims=1, **kwargs):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if isinstance(torch_dtype, str):
        if torch_dtype == 'float16':
            torch_dtype = torch.float16
        elif torch_dtype == 'bfloat16':
            torch_dtype = torch.bfloat16
        elif torch_dtype == 'float32':
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

    kwargs['torch_dtype'] = torch_dtype
    return ModelWithGradiend.from_pretrained(model, layers, activation=activation, activation_decoder=activation_decoder, bias_decoder=bias_decoder, grad_iterations=grad_iterations, decoder_factor=decoder_factor, torch_dtype=torch_dtype, latent_dim=num_dims), kwargs

def train_single_layer_gradiend(model, layer='base_model.encoder.layer.10.output.dense.weight', **kwargs):
    bert_with_ae, kwargs = create_bert_with_ae(model, [layer], **kwargs)
    return train(bert_with_ae, **kwargs)

def train_multiple_layers_gradiend(model, layers, **kwargs):
    bert_with_ae, kwargs = create_bert_with_ae(model, layers, **kwargs)
    return train(bert_with_ae, **kwargs)

def train_all_layers_gradiend(config, model='bert-base-cased', multi_task=False, **kwargs):
    bert_with_ae, kwargs = create_bert_with_ae(model=model, **kwargs)
    return train(bert_with_ae,config=config, multi_task=multi_task, **kwargs)


#does not work that well...
def plot_grad_flow(named_parameters):
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
       
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) 
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
  


if __name__ == '__main__':
    train_all_layers_gradiend('bert-base-cased',
                              output='results/models/gradiend',
                              checkpoints=False,
                              max_iterations=1000000,
                              criterion_ae=nn.MSELoss(),
                              batch_size=8,
                              batch_size_data=None,
                              activation='relu',
                              )
