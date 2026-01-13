import json
import time

import torch
import random
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

from gradiend.model import InstructTokenizerWrapper

import torch
import torch.nn.functional as F
import time
import math
import pandas as pd
import json
import random

def evaluate_clm_perplexity(model, tokenizer, text_data, file=None, verbose=True, batch_size=32):
    print('Evaluating CLM Perplexity')
    model.eval()
    device = model.device
    total_log_likelihood = 0.0
    total_token_count = 0
    stats_data = []

    start = time.time()

    for start_idx in range(0, len(text_data), batch_size):
        end_idx = min(start_idx + batch_size, len(text_data))
        batch_sentences = text_data[start_idx:end_idx]

        if verbose:
            print(f"Processing batch {start_idx + 1}-{end_idx}/{len(text_data)}")

        encoded = tokenizer(batch_sentences, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Shift input and labels for causal LM
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore padding in loss computation

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            # Flatten for loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='sum',  # sum log probs over tokens
                ignore_index=-100
            )

        # Count actual tokens (not padding)
        non_pad_tokens = (labels != -100).sum().item()
        total_log_likelihood += loss.item()
        total_token_count += non_pad_tokens

    avg_neg_log_likelihood = total_log_likelihood / total_token_count
    perplexity = math.exp(avg_neg_log_likelihood)

    if verbose:
        print(f"Evaluated {len(text_data)} sentences in {time.time() - start:.2f}s")
        print(f"Perplexity: {perplexity:.2f}")

    lms = 1 / (1 + avg_neg_log_likelihood)

    result = {
        "accuracy": lms, # todo deprecate
        "perplexity": perplexity,
        "total_log_likelihood": total_log_likelihood,
        "total_tokens": total_token_count
    }

    if file:
        with open(file + '.json', 'w+', encoding='utf8') as f:
            json.dump(result, f, indent=2)

    return result



def evaluate_clm(model, tokenizer, text_data, file=None, verbose=True, batch_size=128):
    random.seed(42)
    model.eval()
    device = model.device
    is_instruction_tokenizer = isinstance(tokenizer, InstructTokenizerWrapper)

    if is_instruction_tokenizer:
        return _evaluate_clm_instruction(model, tokenizer, text_data, file=file, verbose=verbose, batch_size=32)

    correct_predictions = 0
    total_predictions = 0
    true_labels = []
    predicted_labels = []
    stats_data = []

    start = time.time()
    n = len(text_data)


    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_sentences = text_data[start_idx:end_idx]

        if verbose:
            print(f'Processing batch {start_idx + 1}-{end_idx}/{n}')

        batch_tokenized_input = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True,
                                          add_special_tokens=True)
        input_ids = batch_tokenized_input["input_ids"].to(device)
        attention_mask = batch_tokenized_input["attention_mask"].to(device)

        target_positions = []
        for i in range(input_ids.size(0)):
            token_positions = torch.where(attention_mask[i] != 0)[0].tolist()

            if len(token_positions) > 1:
                # use a random token in the 2nd half of the sentence to ensure enough context
                chosen_pos = random.choice(token_positions[len(token_positions)//2:])
                target_positions.append(chosen_pos)
            else:
                target_positions.append(None)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = logits.argmax(dim=-1)

        for i in range(input_ids.size(0)):
            sentence = batch_sentences[i]
            target_pos = target_positions[i]

            if target_pos is not None:
                predicted_token = tokenizer.decode([predictions[i, target_pos]], skip_special_tokens=True)
                # todo check gpt
                true_token = tokenizer.decode([input_ids[i, target_pos]], skip_special_tokens=True)
                correct = predicted_token.lower() == true_token.lower()

                if correct:
                    correct_predictions += 1
                total_predictions += 1

                true_labels.append(true_token.lower())
                predicted_labels.append(predicted_token.lower())

                stats_data.append({
                    'sentence': sentence,
                    'token_index': target_pos,
                    'true': true_token,
                    'predicted': predicted_token,
                    'correct': correct,
                    'score': logits[i, target_pos].max().item()
                })

    stats = pd.DataFrame(stats_data)
    accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    if verbose:
        print(f"Evaluated {n} sentences in {time.time() - start:.2f} seconds")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    if file:
        stats.to_csv(file + '.csv', index=False)
        with open(file + '.json', 'w+', encoding='utf8') as f:
            json.dump(result, f, indent=2)

    return result, stats


def _evaluate_clm_instruction(model, tokenizer, text_data, file=None, verbose=True, batch_size=1):
    device = model.device
    correct_predictions = 0
    total_predictions = 0
    true_labels = []
    predicted_labels = []
    stats_data = []

    start = time.time()
    n = len(text_data)

    end_header_token_id = tokenizer.tokenizer.convert_tokens_to_ids(tokenizer.tokenizer.END)

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_sentences = text_data[start_idx:end_idx]

        if verbose:
            print(f'Processing batch {start_idx + 1}-{end_idx}/{n}')

        batch_tokenized_input = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True,
                                          add_special_tokens=True)
        input_ids = batch_tokenized_input["input_ids"].to(device)
        attention_mask = batch_tokenized_input["attention_mask"].to(device)

        target_positions = []
        for i in range(input_ids.size(0)):
            token_positions = torch.where(attention_mask[i] != 0)[0].tolist()
            ids = input_ids[i].tolist()
            #start_of_user_prompt = len(ids) - 1 - ids[::-1].index(end_header_token_id) + 1
            matches = [j for j, token_id in enumerate(ids) if token_id == end_header_token_id]
            if len(matches) >= 2:
                second_last_index = matches[-2]  # second last occurrence
                start_of_user_prompt = second_last_index + 1
            else:
                start_of_user_prompt = 0

            remaining_token_count = len(token_positions) - start_of_user_prompt
            if remaining_token_count > 1:
                # use a random token in the 2nd half of the user prompt to ensure enough context
                # step 1: determine start of user prompt
                chosen_pos = random.choice(token_positions[start_of_user_prompt + remaining_token_count//2:]) - 1
                target_positions.append(chosen_pos)
            else:
                target_positions.append(None)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = logits.argmax(dim=-1)

        for i in range(input_ids.size(0)):
            sentence = batch_sentences[i]
            target_pos = target_positions[i]

            if target_pos is not None:
                predicted_token = tokenizer.decode([predictions[i, target_pos]], skip_special_tokens=True)
                true_token = tokenizer.decode([input_ids[i, target_pos+1]], skip_special_tokens=True)
                correct = predicted_token.lower() == true_token.lower()

                if correct:
                    correct_predictions += 1
                total_predictions += 1

                true_labels.append(true_token.lower())
                predicted_labels.append(predicted_token.lower())

                stats_data.append({
                    'sentence': sentence,
                    'token_index': target_pos,
                    'true': true_token,
                    'predicted': predicted_token,
                    'correct': correct,
                    'score': logits[i, target_pos].max().item()
                })

        del input_ids
        del attention_mask
        del batch_tokenized_input
        # release GPU memory
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.synchronize()

    stats = pd.DataFrame(stats_data)
    accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    if verbose:
        print(f"Evaluated {n} sentences in {time.time() - start:.2f} seconds")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    if file:
        stats.to_csv(file + '.csv', index=False)
        with open(file + '.json', 'w+', encoding='utf8') as f:
            json.dump(result, f, indent=2)

    return result, stats

def evaluate_mlm(model, tokenizer, text_data, file=None, verbose=True, batch_size=128):
    random.seed(42)
    model.eval()
    device = model.device

    # Initialize variables for accuracy calculation
    correct_predictions = 0
    total_predictions = 0
    true_labels = []
    predicted_labels = []
    stats_data = []

    start = time.time()
    n = len(text_data)
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_sentences = text_data[start_idx:end_idx]

        if verbose:
            print(f'Processing batch {start_idx + 1}-{end_idx}/{n}')

        # Tokenize the batch
        batch_tokenized_input = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True,
                                          add_special_tokens=True)
        input_ids = batch_tokenized_input["input_ids"].to(device)
        attention_mask = batch_tokenized_input["attention_mask"].to(device)

        # Randomly mask tokens in the batch
        mask_positions_batch = []
        masked_input_ids_batch = input_ids.clone()

        for i in range(input_ids.size(0)):
            token_positions = torch.where(
                (input_ids[i] != tokenizer.all_special_ids) & (attention_mask[i] != 0)
            )[0].tolist()

            # Masking tokens based on a 15% probability
            num_to_mask = max(1, int(0.15 * len(token_positions)))
            mask_positions = random.sample(token_positions, num_to_mask)
            masked_input_ids_batch[i, mask_positions] = tokenizer.mask_token_id
            mask_positions_batch.append(mask_positions)

        # Perform MLM predictions in batch
        with torch.no_grad():
            outputs = model(input_ids=masked_input_ids_batch, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = logits.argmax(dim=-1)

        # Collect stats in batch
        for i in range(input_ids.size(0)):
            sentence = batch_sentences[i]
            original_tokens = input_ids[i]
            predicted_tokens_batch = predictions[i]
            mask_positions = mask_positions_batch[i]
            logits_for_sentence = logits[i]

            # Process each masked position only
            for mask_position in mask_positions:
                predicted_token = tokenizer.decode([predicted_tokens_batch[mask_position]],
                                                   skip_special_tokens=True)
                original_token = tokenizer.decode([original_tokens[mask_position]], skip_special_tokens=True)
                if not original_token.strip():
                    continue

                correct = predicted_token.lower() == original_token.lower()

                if correct:
                    correct_predictions += 1
                total_predictions += 1

                # Append to labels for evaluation
                true_labels.append(original_token.lower())
                predicted_labels.append(predicted_token.lower())

                stats_data.append({
                    'sentence': sentence,
                    'token_index': mask_position,
                    'true': original_token,
                    'predicted': predicted_token,
                    'correct': correct,
                    'score': logits_for_sentence[mask_position].max().item()
                })

    # Convert stats_data into DataFrame at the end to minimize DataFrame operations in the loop
    stats = pd.DataFrame(stats_data)

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0

    # Calculate additional evaluation metrics
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    # Print results
    if verbose:
        print(f"Evaluated {n} sentences in {time.time() - start:.2f} seconds")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    if file:
        stats.to_csv(file + '.csv', index=False)
        with open(file + '.json', 'w+', encoding='utf8') as f:
            json.dump(result, f, indent=2)

    return result, stats

def get_top_k_predictions(model, tokenizer, masked_phrase: str, k: int = 5):
    # Tokenize the input
    inputs = tokenizer(masked_phrase, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get the mask token index
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]

    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # Get the top k token predictions for the masked token
    mask_token_logits = logits[0, mask_token_index, :]
    top_k_logits = torch.topk(mask_token_logits, k, dim=1).values
    top_k_indices = torch.topk(mask_token_logits, k, dim=1).indices

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(top_k_logits, dim=1)

    # Prepare the results dictionary
    results = {}
    for i in range(k):
        token_id = top_k_indices[0, i].item()
        token = tokenizer.decode([token_id])
        probability = probabilities[0, i].item()
        results[token] = probability

    return results