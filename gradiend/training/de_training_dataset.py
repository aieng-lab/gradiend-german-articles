from collections import defaultdict, deque
import glob
import os
import random
import time
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from gradiend.data import read_article_ds
from gradiend.util import hash_it
from torch.utils.data.sampler import Sampler
from torch.utils.data import default_collate
from torch.utils.data import Sampler
from collections import defaultdict, deque
import random


class DeTrainingDataset(Dataset):
    def __init__(self, data, config, tokenizer, batch_size, max_token_length=None, is_generative=False, multi_task=False):
        self.data = data
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.max_token_length = max_token_length or (256 if 'llama' in tokenizer.name_or_path.lower() else 128)
        self.config = config
        self.multi_task = multi_task

        self.mask_token_id = self.tokenizer.mask_token_id
        self.mask_token = self.tokenizer.mask_token

        articles = self.config['articles']

        self.article_tokens = {(determiner, upper): self.tokenizer.encode(
            determiner[0].upper() + determiner[1:] if upper else determiner,
            add_special_tokens=False)[0] for determiner in articles for upper in [True, False]}


    """Returns the number of entries in the datasets"""
    def __len__(self):
        return len(self.data)


    '''Returns the tokenIds and the attention masks as a torch tensor.'''
    def tokenize(self, text):
        if self.tokenizer.mask_token:
            item = self.tokenizer(text, truncation=True, padding='max_length',
                                  max_length=self.max_token_length, return_tensors="pt")
        else:
            item = self.tokenizer(text, truncation=True, padding='max_length',
                                  max_length=self.max_token_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in item.items()}
        return item

 
    def __getitem__(self, index):
        entry = self.data.iloc[index]
        is_gender_data=True
        
        if is_gender_data:
            masked_entry = entry['masked']
           
            key = entry['dataset_label']
            determiner = entry['label'].lower()
            inverse_determiner = self.config[key]['inverse']
         
            def fill(text):
                if self.mask_token:
                    return text.replace(self.config[key]['mask'], self.mask_token)
                split = text.split(self.config[key]['mask'])
                return split[0]


            gender_text = fill(masked_entry)
    
            item = self.tokenize(gender_text)
            gender_labels = item['input_ids'].clone()
            

            if self.multi_task: 
                inv_labels = []
                counterfactuals = len(inverse_determiner)
                for i in range(0, counterfactuals): 
                    inv_labels.append(gender_labels.clone())
            else:
                inv_gender_labels = gender_labels.clone()


            sentence_delimiter = {'.', '!', '?'}
        
            mask_token_mask = gender_labels == self.mask_token_id

            gender_text_no_white_spaces = gender_text.replace(' ', '')

            if self.mask_token:
                if self.mask_token not in gender_text_no_white_spaces:
                    print("mask_index not found for",  entry['masked'])


                mask_index = gender_text_no_white_spaces.index(self.mask_token)
                upper = mask_index == 0 or mask_index > 2 and gender_text_no_white_spaces[
                    mask_index - 1] in sentence_delimiter and gender_text_no_white_spaces[mask_index - 2] != '.'

                # only compute loss on masked tokens
                gender_labels[~mask_token_mask] = -100
                gender_labels[mask_token_mask] = self.article_tokens[(
                    determiner, upper)]

                if self.multi_task:
                    for inv_label, inv_det in zip(inv_labels, inverse_determiner):
                        inv_label[~mask_token_mask] = -100
                        inv_label[mask_token_mask] = self.article_tokens[(inv_det, upper)]
                else:
                    inv_gender_labels[~mask_token_mask] = -100
                    inv_gender_labels[mask_token_mask] = self.article_tokens[(
                    inverse_determiner, upper)]
            else:
                # no mask token,
                pass

            if not self.multi_task:
                inv_item = item.copy()

            item['labels'] = gender_labels
            label = self.config[key]['code'] 
            text = gender_text

            if self.multi_task:
                # entries = []
                inv_items = []
                for inv_label in inv_labels: 
                    inv_item = item.copy() 
                    inv_item['labels'] = inv_label
                    inv_items.append(inv_item)

                entry = {
                    True: item,       
                    False: inv_items,       
                    'text': text,
                    'label': label,
                    'dataset_label': key
                    }
                    
                # entries.append(entry)       
            else:
                inv_item['labels'] = inv_gender_labels
    
        else:
            text = entry['text']
            item = self.tokenize(text)
            masked_input, labels = self.mask_tokens(item['input_ids'])
            item['input_ids'] = masked_input
            item['labels'] = labels
            inv_item = item
            label = ''

        if self.multi_task: 
            return entry
        else:
            return {True: item, False: inv_item, 'text': text, 'label': label, 'dataset_label': key, 'inverse': inverse_determiner, 'determiner': determiner}


# def flatten_dict_list_collate(batch, multi_grad=True):
#     if multi_grad: 
#         flat_batch = [sample for sublist in batch for sample in sublist]
#     else:
#         flat_batch = batch

#     print(type(batch))

#     return default_collate(flat_batch)


def flatten_dict_list_collate(batch, multi_grad=True):
    if multi_grad and all(isinstance(sample, list) for sample in batch):
        flat_batch = [sample for sublist in batch for sample in sublist]
    else:
        flat_batch = batch

    if isinstance(flat_batch[0], dict):
        return {
            key: _safe_collate([d[key] for d in flat_batch])
            for key in flat_batch[0]
        }
    return default_collate(flat_batch)


def _safe_collate(values):
    try:
        return default_collate(values)
    except Exception:
        return values 


# def flatten_dict_list_collate(batch, multi_grad=True):
#     """
#     batch: List of dicts returned by __getitem__, each like:
#         {
#           True: {...},
#           False: {...},
#           'text': ...,
#           'label': ...,
#           'dataset_label': ...
#         }

#     If multi_grad=True, the outer list may be nested (i.e. list of lists of dicts),
#     so we flatten that first.
#     """
#     # Flatten if each sample is itself a list of entries
#     if multi_grad:
#         flat_batch = [entry for sublist in batch for entry in sublist]
#     else:
#         flat_batch = batch

#     # Now group by key
#     collated = {}
#     keys = flat_batch[0].keys()

#     for key in keys:
#         values = [entry[key] for entry in flat_batch]
#         try:
#             collated[key] = default_collate(values)
#         except Exception as e:
#             print(f"Collate failed for key '{key}' with error: {e}")
#             collated[key] = values  # Fall back to keeping as list if needed

#     return collated


def create_de_training_dataset(tokenizer, config, max_size=None, batch_size=None, split=None, article=None, is_generative=False, multi_task=False):

    dataset = read_article_ds(split=split, article=article)
    if max_size:
        dataset = dataset.iloc[range(max_size)]

    return DeTrainingDataset(dataset, config, tokenizer, batch_size=batch_size, is_generative=is_generative, multi_task=multi_task)



class FlattenedConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.flat_items = []
        for ds in datasets:
            for i in range(len(ds)):
                item = ds[i]
                if isinstance(item, list):
                    self.flat_items.extend(item)
                else:
                    self.flat_items.append(item)

    def __getitem__(self, idx):
        return self.flat_items[idx]

    def __len__(self):
        return len(self.flat_items)


def create_de_eval_dataset(gradiend, config, max_size=None, split='val', source='gradient', save_layer_files=False, is_generative=False, multi_task=False, save_gradients=False):
    if not source in {'gradient', 'inv_gradient', 'diff'}:
        raise ValueError(f'Invalid source {source}')

    start = time.time()

    eval_datasets = []
    for label in config['combinations']:
        eval_dataset = create_de_training_dataset(gradiend.tokenizer, config, article=label, split=split, multi_task=multi_task)
        eval_datasets.append(eval_dataset)

    dataset = CombinedDataset(eval_datasets)

    #texts = datasets.data.loc[:, ['masked', 'label', 'dataset_label']]
    
    texts = dataset.data.sample(frac=1, random_state=42).reset_index(drop=True)
    #texts = shuffled.loc[:max_size, ['masked', 'label']]
    if max_size:
        if 0.0 <= max_size <= 1.0:
            max_size = int(max_size * len(texts))
            print('eval max_size', max_size)
        texts = texts[:(max_size)]

    mask_token = gradiend.tokenizer.mask_token

   
    filled_texts = {}
    for _, row in texts.iterrows():
        text = row['masked']
        key = row['dataset_label']
       
        filled_text = text.replace(config[key]['mask'], mask_token)
        filled_texts[filled_text] = {"label": row['label'], "dataset_label": row['dataset_label']}


    # calculate the gradients in advance, if not already cached?
    base_model = gradiend.base_model.name_or_path
    base_model = os.path.basename(base_model)
    layers_hash = gradiend.layers_hash
    cache_dir = f'data/cache/gradients/{base_model}/{source}/{layers_hash}'

    if multi_task: 
        gradients = defaultdict(lambda: defaultdict(list))
    else:
        gradients = defaultdict(dict)  # maps texts to the gradients
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # the actual evaluation data being loaded and the gradients being calculated
    text_iterator = tqdm(
        filled_texts, desc=f'Loading cached evaluation data', leave=False)

    for i, filled_text in enumerate(text_iterator):
        text_hash = hash_it(filled_text)

        def create_layer_file(layer, index=None):
            filename = f'{layer}.pt' if index is None else f'{layer}__{index}.pt'
            return f'{cache_dir}/{text_hash}/{filename}'
        

        if multi_task: 
            pattern = f'{cache_dir}/tensor_{text_hash}_*.pt'
            cached_files = sorted(glob.glob(pattern))  

            if cached_files:
                gradients[filled_text]['full'] = []
                for file in cached_files:
                    gradient = torch.load(file).half().cpu()
                    gradients[filled_text]['full'].append(gradient)
                continue
        else:
            cached_tensor_file = f'{cache_dir}/tensor_{text_hash}.pt'
            if os.path.exists(cached_tensor_file):
                gradient = torch.load(cached_tensor_file).half().cpu()
                gradients[filled_text] = gradient
                continue

        # first check whether we need to calculate the gradients
        requires_grad = any(not os.path.exists(create_layer_file(layer))
                            for layer in gradiend.gradiend.layers)

        # only compute the gradients (computationally expensive) if really needed
        if requires_grad:
            print(f"Calculate gradients for {filled_text} and label {filled_texts[filled_text]['label']}")

            label = filled_texts[filled_text]['label'].lower()
            inv_label = config[filled_texts[filled_text]['dataset_label']]['inverse']

            if source == 'diff':
                label_factual = label
                label_counter_factual = inv_label

                inputs_factual = gradiend.create_inputs(
                    filled_text, label_factual)
                grads_factual = gradiend.forward_pass(
                    inputs_factual, return_dict=True)
                inputs_counter_factual = gradiend.create_inputs(
                    filled_text, label_counter_factual)
                grads_counter_factual = gradiend.forward_pass(
                    inputs_counter_factual, return_dict=True)
                grads = {layer: grads_factual[layer] - grads_counter_factual[layer]
                         for layer in gradiend.gradiend.layers}
            else:
                if source == 'gradient':
                    label = label
                elif source == 'inv_gradient':
                    label = inv_label
                    #config[filled_texts[filled_text]['dataset_label']]['inverse']

                if multi_task:
                    all_grads = [] 
                    for label in inv_label: 
                        inputs = gradiend.create_inputs(filled_text, label)
                        grads = gradiend.forward_pass(inputs, return_dict=True)
                        all_grads.append(grads)
                else:
                    inputs = gradiend.create_inputs(filled_text, label)
                    grads = gradiend.forward_pass(inputs, return_dict=True)

            if save_layer_files:
                # create the directory
                dummy_file = create_layer_file('dummy')
                os.makedirs(os.path.dirname(dummy_file), exist_ok=True)
        else:
            grads = None

        for layer in gradiend.gradiend.layers:
            if multi_task: 
                for idx, grad in enumerate(all_grads):
                    layer_file = create_layer_file(layer, index=idx)
                    if not os.path.exists(layer_file):
                        weights = grad[layer].half().flatten().cpu()
                    else: 
                        weights = torch.load(layer_file, weights_only=False)
                    gradients[filled_text][layer].append(weights.float())    
            else:         
                layer_file = create_layer_file(layer, index=None)
                if not os.path.exists(layer_file):
                    weights = grads[layer].half().flatten().cpu()
       
                    if save_layer_files:
                        # Saving individual layer files doubles the necessary storage, but is more efficient when working with different layer subsets
                        torch.save(weights, layer_file)
                else:
                    weights = torch.load(layer_file, weights_only=False)

                weights = weights.float()
                gradients[filled_text][layer] = weights


            # gradients[filled_text][layer] = weights

        # convert layer dict to single tensor
        if multi_task: 
            num_grads = len(next(iter(gradients[filled_text].values())))
            gradients[filled_text]['full'] = []

         
            for idx in range(num_grads):
                full_gradient = torch.concat([gradients[filled_text][layer][idx] for layer in gradients[filled_text] if layer != 'full'], dim=0
                ).half()

                gradient = full_gradient
                if isinstance(gradiend.gradiend.layers, dict):
                    mask = torch.concat([gradiend.gradiend.layers[k].flatten() for k in gradients[filled_text].keys()], dim=0).cpu()
                    gradient = full_gradient[mask]
                
                gradients[filled_text]['full'].append(gradient)
                cached_tensor_file = f'{cache_dir}/tensor_{text_hash}_{idx}.pt' 
                #cache_file = os.path.join(cache_dir, f"{text_hash}__{idx}_full.pt")
                if save_gradients:
                    torch.save(gradient, cached_tensor_file)
            
        else:
            full_gradient = torch.concat(
                [v for v in gradients[filled_text].values()], dim=0).half()
            gradient = full_gradient
            if isinstance(gradiend.gradiend.layers, dict):
                mask = torch.concat(
                    [gradiend.gradiend.layers[k].flatten() for k in gradients[filled_text].keys()], dim=0
                ).cpu()
                gradient = full_gradient[mask]

            gradients[filled_text] = gradient
            if save_gradients:
                torch.save(gradient, cached_tensor_file)


    return create_grads_and_labels(filled_texts, config, multi_grad=multi_task, gradients=gradients)


def create_grads_and_labels(filled_texts, config, multi_grad, gradients): 
    labels = {}
    flat_gradients = {}
    encodings = {}

    label_code_to_encoding = {v['code']: v['encoding'] for k,v in config.items() if 'code' in v and 'encoding' in v}

    for filled_text, v in filled_texts.items():
        label_code = config[v['dataset_label']]['code']

        if multi_grad:
            for idx, grad in enumerate(gradients[filled_text]['full']):
                key = f"{filled_text}_{idx}"
                flat_gradients[key] = grad
                labels[key] = label_code
                encodings[key] = label_code_to_encoding[label_code]
        else:
            flat_gradients[filled_text] = gradients[filled_text]
            labels[filled_text] = label_code
            encodings[filled_text] = label_code_to_encoding[label_code]

    result = {
        'gradients': flat_gradients,
        'labels': labels,
        'encodings': encodings,
        }
    
    print(
        f'Loaded the evaluation data with {len(gradients)} entries')
    
    return result


class DynamicEvalDataset:
    def __init__(self, gradiend, config, split='val', source='gradient', max_size=None, multi_task=False):
        self.gradiend = gradiend
        self.config = config
        self.source = source
        self.multi_task = multi_task

        eval_datasets = []
        for label in config['combinations']:
            ds = create_de_training_dataset(gradiend.tokenizer, config, article=label, split=split, multi_task=multi_task)
            eval_datasets.append(ds)

        combined = CombinedDataset(eval_datasets)
        texts = combined.data.sample(frac=1, random_state=42).reset_index(drop=True)

        if max_size:
            if 0.0 <= max_size <= 1.0:
                max_size = int(max_size * len(texts))
            texts = texts[:max_size]

        mask = gradiend.tokenizer.mask_token
        self.items = []
        for _, row in texts.iterrows():
            key = row['dataset_label']
            filled = row['masked'].replace(config[key]['mask'], mask)
            entry = {
                'text': filled,
                'label': row['label'],
                'dataset_label': key
            }
            self.items.append(entry)

    def __len__(self):
        return len(self.items)

    def _compute_grad(self, text, dataset_label, label):
        inv_label = self.config[dataset_label]['inverse']

        if self.source == 'diff':
            factual = label.lower()
            counter = inv_label
            inp_f = self.gradiend.create_inputs(text, factual)
            inp_c = self.gradiend.create_inputs(text, counter)
            gf = self.gradiend.forward_pass(inp_f, return_dict=True)
            gc = self.gradiend.forward_pass(inp_c, return_dict=True)
            return {layer: gf[layer] - gc[layer] for layer in self.gradiend.gradiend.layers}

        if self.source == 'inv_gradient':
            label = inv_label
        else:
            label = label.lower()

        if self.multi_task:
            all_grads = []
            for lab in inv_label:
                inp = self.gradiend.create_inputs(text, lab)
                g = self.gradiend.forward_pass(inp, return_dict=True)
                all_grads.append(g)
            return all_grads

        inp = self.gradiend.create_inputs(text, label)
        return self.gradiend.forward_pass(inp, return_dict=True)

    def _flatten(self, grads):
        if self.multi_task:
            out = []
            for g in grads:
                flat = torch.concat([g[layer].half().flatten() for layer in g], dim=0)
                out.append(flat)
            return out

        flat = torch.concat([grads[layer].half().flatten() for layer in grads], dim=0)
        return flat

    def __getitem__(self, idx):
        item = self.items[idx]
        text = item['text']
        label = item['label']
        ds_label = item['dataset_label']

        grads = self._compute_grad(text, ds_label, label)
        flat = self._flatten(grads)

        code = self.config[ds_label]['code']
        encoding = self.config[ds_label]['encoding']

        if self.multi_task:
            return {
                'text': text,
                'label': code,
                'encoding': encoding,
                'grad': flat
            }

        return {
            'text': text,
            'label': code,
            'encoding': encoding,
            'grad': flat
        }


def create_de_eval_dataset_no_cache(model_with_gradiend, config, split, source, eval_max_size, is_generative, multi_task):
    eval_dataset = DynamicEvalDataset(
        gradiend=model_with_gradiend,
        config=config,
        split=split,
        source=source,
        max_size=eval_max_size,
        multi_task=multi_task,
    )
    return eval_dataset


class SingleLabelBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False): 
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
    
        self.labels = defaultdict(list)

        for idx in range(len(dataset)):
            label = dataset[idx]['dataset_label']
            self.labels[label].append(idx)

        self.batches = self._create_batches()
        print('#labels in the Sampler', len(self.labels))


    def _create_batches(self): 
        label_batches = {}
        for label, indices in self.labels.items():
            label_batches[label] = deque()
            for i in range(0, len(indices), self.batch_size): 
                batch = indices[i:i+ self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                
                label_batches[label].append(batch)

        # batches are interleaved, so it goes label 1, label 2, label 1 etc... 
        interleaved = []   
        label_cycle = list(label_batches.keys())


        while any(label_batches.values()):
            for label in label_cycle: 
                if label_batches[label]:
                    batch = label_batches[label].popleft()
                    interleaved.append(batch)


        return interleaved         
             

    def __iter__(self):
        for batch in self.batches: 
            yield batch
    

    def __len__(self): 
        return len(self.batches)                  
                    

class OversampledSingleLabelBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed

        self.labels = defaultdict(list)
        for idx in range(len(dataset)):
            label = dataset[idx]["dataset_label"]
            self.labels[label].append(idx)

        self.batches = self._create_batches()
        print("#labels in sampler:", len(self.labels))

    def _make_label_batches(self, indices):
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)
        return batches

    def _create_batches(self):
        rng = random.Random(self.seed)

        # batches per label
        label_batches = {}
        max_batches = 0

        for label, indices in self.labels.items():
            rng.shuffle(indices)
            batches = self._make_label_batches(indices)
            label_batches[label] = deque(batches)
            max_batches = max(max_batches, len(batches))

        # oversample to max_batches
        for label, batches in label_batches.items():
            if len(batches) == 0:
                continue

            while len(batches) < max_batches:
                batches.append(rng.choice(list(batches)))

        # round-robin interleaving
        interleaved = []
        label_cycle = list(label_batches.keys())

        for i in range(max_batches):
            for label in label_cycle:
                interleaved.append(label_batches[label].popleft())

        return interleaved

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)



class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.data = pd.concat([dataset.data for dataset in self.datasets], ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]  
    

class PairedLabelBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            label = dataset[idx]['dataset_label']
            self.label_to_indices[label].append(idx)

        # Group labels into pairs (NM + _NM)
        self.paired_labels = self._create_label_pairs()
        self.batches = self._create_batches()
        print(f'# paired label groups: {len(self.paired_labels)}')

    def _create_label_pairs(self):
        pairs = defaultdict(list)
        for label in self.label_to_indices:
            base_label = label.replace('_', '')  # e.g., NM from _NM
            pairs[base_label].append(label)
        return pairs

    def _create_batches(self):
        paired_batches = []
        for base_label, label_group in self.paired_labels.items():
            if len(label_group) != 2:
                print(f"incomplete pair for label: {label_group}")
                continue

            label1, label2 = label_group
            indices1 = self.label_to_indices[label1]
            indices2 = self.label_to_indices[label2]

           
            random.shuffle(indices1)
            random.shuffle(indices2)

            split_size = self.batch_size // 2
            min_len = min(len(indices1), len(indices2))

            max_batches = min_len // split_size

            for i in range(max_batches):
                part1 = indices1[i * split_size:(i + 1) * split_size]
                part2 = indices2[i * split_size:(i + 1) * split_size]
                batch = part1 + part2
                if len(batch) == self.batch_size or not self.drop_last:
                    paired_batches.append(batch)

        return paired_batches

    def __iter__(self):
        random.seed(42)
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
