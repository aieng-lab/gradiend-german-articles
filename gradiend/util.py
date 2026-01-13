import ast
import io
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
from matplotlib import font_manager as fm, pyplot as plt

default_accuracy_function = lambda x: x #np.power(x, 10)
normalization = lambda x: x #np.power(x, 0.1)

case_gender_mapping = {
    'N': {'F': 'die', 'M': 'der', 'N': 'das', 'P': 'die'}, # Nominativ
    'G': {'F': 'der', 'M': 'des', 'N': 'des', 'P': 'der'}, # Genitiv
    'D': {'F': 'der', 'M': 'dem', 'N': 'dem', 'P': 'den'}, # Dativ
    'A': {'F': 'die', 'M': 'den', 'N': 'das', 'P': 'die'}, # Akkusativ
}

RESULTS_DIR = 'results'
IMAGE_DIR = 'img'

def init_matplotlib(font_path="/root/times.ttf", use_tex=False):
    # Check if the font is already in the font manager
    for font in fm.fontManager.ttflist:
        if font.fname == font_path:
            print(f"Font at {font_path} is already loaded.")
            plt.rcParams['font.family'] = font.name
            return

    font_entry = fm.FontEntry(
        fname=font_path,
        name="CustomFont", 
        style="normal",
        weight="normal",
        stretch="normal"
    )

    # Append to Matplotlib's font manager
    fm.fontManager.ttflist.append(font_entry)
    plt.rcParams['font.family'] = 'CustomFont'

    if use_tex:

        preamble = r"""
% --- Core packages (keep) ---
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}

% --- Naming macros (text mode) ---
\newcommand{\namexact}{\textsc{NAMExact}}
\newcommand{\namextend}{\textsc{NAMExtend}}
\newcommand{\gradiend}{\textsc{Gradiend}}
\newcommand{\dropout}{\textsc{Dropout}}
\newcommand{\selfdebias}{\textsc{SelfDebias}}
\newcommand{\sentencedebias}{\textsc{SentDebias}}
\newcommand{\inlp}{\textsc{INLP}}
\newcommand{\cda}{\textsc{CDA}}

\newcommand{\genter}{\textsc{Genter}}
\newcommand{\geneutral}{\textsc{GENeutral}}
\newcommand{\gentypes}{\textsc{GENTypes}}

% --- Safe math helpers (always use ensuremath; never embed $...$ inside macros) ---
\newcommand{\namexacttrain}{\ensuremath{\textsc{NAMExact}_{\text{train}}}}
\newcommand{\namexacttest}{\ensuremath{\textsc{NAMExact}_{\text{test}}}}

\newcommand{\gentertrain}{\ensuremath{\genter_{\text{train}}}}
\newcommand{\gentertest}{\ensuremath{\genter_{\text{test}}}}
\newcommand{\genterval}{\ensuremath{\genter_{\text{val}}}}
\newcommand{\genterzero}{\ensuremath{\genter^{0}}}

% --- Model names (math-safe) ---
\newcommand{\bertbase}{\ensuremath{\text{BERT}_{\text{base}}}}
\newcommand{\bertlarge}{\ensuremath{\text{BERT}_{\text{large}}}}
\newcommand{\roberta}{\textsc{RoBERTa}}
\newcommand{\distilbert}{\textsc{DistilBERT}}
\newcommand{\gpttwo}{\textsc{GermanGPT-2}}
\newcommand{\llama}{\textsc{LLaMA}}
% NOTE: the original \llamai used "\llama-Instruct" which is not a valid macro token.
% Use text instead:
\newcommand{\llamai}{\textsc{LLaMA-Instruct}}

% --- Metrics (math-safe) ---
\newcommand{\acc}{\ensuremath{\text{Acc}}}
\newcommand{\cor}{\ensuremath{\text{Cor}}}
\newcommand{\enc}{\ensuremath{\text{Enc}}}
\newcommand{\dec}{\ensuremath{\text{Dec}}}

\newcommand{\accenc}{\ensuremath{\acc_{\enc}}}
\newcommand{\accdec}{\ensuremath{\acc_{\dec}}}
\newcommand{\corenc}{\ensuremath{\cor_{\enc}}}

\newcommand{\cormf}{\ensuremath{\cor_{\text{\genter}}}}
\newcommand{\cormfval}{\ensuremath{\cor_{\text{\genter}_{\text{val}}}}}
\newcommand{\cormftest}{\ensuremath{\cor_{\text{\genter}_{\text{test}}}}}

\newcommand{\accmf}{\ensuremath{\acc_{\text{\genter}}}}
\newcommand{\mamf}{\ensuremath{\overline{|h|}_{\text{\genter}}}}
\newcommand{\masmf}{\ensuremath{\overline{|h|}_{\text{\genterzero}}}}
\newcommand{\man}{\ensuremath{\overline{|h|}_{\text{\geneutral}}}}

% --- FPI/MPI/BPI (math-safe) ---
\newcommand{\fpi}{\text{FPI}}
\newcommand{\mpi}{\text{MPI}}
\newcommand{\bpi}{\text{BPI}}

\newcommand{\gradiendfpi}{\ensuremath{\text{\gradiend}_{\text{\fpi}}}}
\newcommand{\gradiendmpi}{\ensuremath{\text{\gradiend}_{\text{\mpi}}}}
\newcommand{\gradiendbpi}{\ensuremath{\text{\gradiend}_{\text{\bpi}}}}

% --- Dataset symbol (math-safe) ---
\newcommand{\dataNEUT}{\ensuremath{D_{\textsc{Neutral}}}}

% --- Model identifiers (text mode) ---
\newcommand{\bert}{\textsc{GermanBERT}}
\newcommand{\modernbert}{\textsc{ModernGBERT}}
\newcommand{\gbert}{\textsc{GBERT}}
\newcommand{\eurobert}{\textsc{EuroBERT}}

% --- Cases / gender tags (math-safe) ---
\newcommand{\nominative}{\ensuremath{\textsc{Nom}}}
\newcommand{\accusative}{\ensuremath{\textsc{Acc}}}
\newcommand{\dative}{\ensuremath{\textsc{Dat}}}
\newcommand{\genitive}{\ensuremath{\textsc{Gen}}}

\newcommand{\male}{\ensuremath{\textsc{Masc}}}
\newcommand{\female}{\ensuremath{\textsc{Fem}}}
\newcommand{\neutral}{\ensuremath{\textsc{Neut}}}

% --- Gradiend symbol (math-safe) ---
\newcommand{\gradiendsymb}{\ensuremath{G}}

% --- ALL gradc* macros: NEVER use $...$ inside; always ensuremath ---
\newcommand{\gradcNgMF}{\ensuremath{\gradiendsymb_{\nominative}^{\female,\male}}}
\newcommand{\gradcNDgF}{\ensuremath{\gradiendsymb_{\nominative,\dative}^{\female}}}
\newcommand{\gradcNGgF}{\ensuremath{\gradiendsymb_{\nominative,\genitive}^{\female}}}
\newcommand{\gradcDAgF}{\ensuremath{\gradiendsymb_{\accusative,\dative}^{\female}}}
\newcommand{\gradcADgF}{\ensuremath{\gradiendsymb_{\accusative,\dative}^{\female}}}
\newcommand{\gradcGAgF}{\ensuremath{\gradiendsymb_{\accusative,\genitive}^{\female}}}
\newcommand{\gradcNDgM}{\ensuremath{\gradiendsymb_{\nominative,\dative}^{\male}}}
\newcommand{\gradcDgMF}{\ensuremath{\gradiendsymb_{\dative}^{\female,\male}}}
\newcommand{\gradcDgFN}{\ensuremath{\gradiendsymb_{\dative}^{\female,\neutral}}}
\newcommand{\gradcNGgM}{\ensuremath{\gradiendsymb_{\nominative,\genitive}^{\male}}}
\newcommand{\gradcGgFM}{\ensuremath{\gradiendsymb_{\genitive}^{\female,\male}}}
\newcommand{\gradcGgMF}{\ensuremath{\gradiendsymb_{\genitive}^{\female,\male}}}
\newcommand{\gradcGgFN}{\ensuremath{\gradiendsymb_{\genitive}^{\female,\neutral}}}

\newcommand{\gradcNgFN}{\ensuremath{\gradiendsymb_{\nominative}^{\female,\neutral}}}
\newcommand{\gradcAgFN}{\ensuremath{\gradiendsymb_{\accusative}^{\female,\neutral}}}
\newcommand{\gradcNDgN}{\ensuremath{\gradiendsymb_{\nominative,\dative}^{\neutral}}}
\newcommand{\gradcADgN}{\ensuremath{\gradiendsymb_{\accusative,\dative}^{\neutral}}}
\newcommand{\gradcNGgN}{\ensuremath{\gradiendsymb_{\nominative,\genitive}^{\neutral}}}
\newcommand{\gradcAGgN}{\ensuremath{\gradiendsymb_{\accusative,\genitive}^{\neutral}}}
\newcommand{\gradcGAgN}{\ensuremath{\gradiendsymb_{\accusative,\genitive}^{\neutral}}}

\newcommand{\gradcNAgM}{\ensuremath{\gradiendsymb_{\nominative,\accusative}^{\male}}}
\newcommand{\gradcAgMF}{\ensuremath{\gradiendsymb_{\accusative}^{\female,\male}}}
\newcommand{\gradcNgMN}{\ensuremath{\gradiendsymb_{\nominative}^{\male,\neutral}}}
\newcommand{\gradcAgMN}{\ensuremath{\gradiendsymb_{\accusative}^{\male,\neutral}}}
\newcommand{\gradcADgM}{\ensuremath{\gradiendsymb_{\accusative,\dative}^{\male}}}

% --- Data symbol macros (math-safe) ---
\newcommand{\datasymbol}{\ensuremath{D}}
\newcommand{\dataNM}{\ensuremath{\datasymbol_{\nominative}^{\male}}}
\newcommand{\dataNF}{\ensuremath{\datasymbol_{\nominative}^{\female}}}
\newcommand{\dataNN}{\ensuremath{\datasymbol_{\nominative}^{\neutral}}}
\newcommand{\dataDM}{\ensuremath{\datasymbol_{\dative}^{\male}}}
\newcommand{\dataDF}{\ensuremath{\datasymbol_{\dative}^{\female}}}
\newcommand{\dataDN}{\ensuremath{\datasymbol_{\dative}^{\neutral}}}
\newcommand{\dataAM}{\ensuremath{\datasymbol_{\accusative}^{\male}}}
\newcommand{\dataAF}{\ensuremath{\datasymbol_{\accusative}^{\female}}}
\newcommand{\dataAN}{\ensuremath{\datasymbol_{\accusative}^{\neutral}}}
\newcommand{\dataGM}{\ensuremath{\datasymbol_{\genitive}^{\male}}}
\newcommand{\dataGF}{\ensuremath{\datasymbol_{\genitive}^{\female}}}
\newcommand{\dataGN}{\ensuremath{\datasymbol_{\genitive}^{\neutral}}}

        """


        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': preamble
        })

def get_tensor_memory_size(tensor):
    return tensor.element_size() * tensor.numel()


def convert_tuple_keys_recursively(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Convert tuple keys to JSON strings (of lists)
            if isinstance(k, tuple):
                k = json.dumps(k)
            new_dict[k] = convert_tuple_keys_recursively(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_tuple_keys_recursively(item) for item in obj]
    else:
        return obj

def restore_tuple_keys_recursively(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            try:
                k_loaded = json.loads(k)
                if isinstance(k_loaded, list):
                    k = tuple(k_loaded)
            except (ValueError, json.JSONDecodeError):
                pass
            new_dict[k] = restore_tuple_keys_recursively(v)
        return new_dict
    elif isinstance(obj, list):
        return [restore_tuple_keys_recursively(item) for item in obj]
    else:
        return obj


def get_total_memory_usage(data):
    total_size = 0
    if isinstance(data, dict):
        for value in data.values():
            total_size += get_total_memory_usage(value)
    elif isinstance(data, torch.Tensor):
        total_size += get_tensor_memory_size(data)
    elif isinstance(data, list):
        for item in data:
            total_size += get_total_memory_usage(item)
    # Add other iterable types if needed
    return total_size

def recursive_gpu_size(obj, seen=None):
    if seen is None:
        seen = set()
    size = 0
    if id(obj) in seen:
        return 0
    seen.add(id(obj))

    if torch.is_tensor(obj):
        if obj.is_cuda:
            size += obj.element_size() * obj.nelement()
    elif isinstance(obj, dict):
        for v in obj.values():
            size += recursive_gpu_size(v, seen)
        for k in obj.keys():
            size += recursive_gpu_size(k, seen)
    elif hasattr(obj, '__dict__'):
        size += recursive_gpu_size(vars(obj), seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        for i in obj:
            size += recursive_gpu_size(i, seen)

    return size

def get_gpu_usage_gb(obj):
    return recursive_gpu_size(obj) / (1024 ** 3)


def sanitize_filename(filename):
    """
    Removes forbidden characters from a string to make it a valid filename.

    Parameters:
    filename (str): The original filename string.

    Returns:
    str: The sanitized filename string.
    """
    # Define the forbidden characters for filenames
    forbidden_chars = r'[\/:*?"<>|]'

    # Replace forbidden characters with an underscore
    sanitized = re.sub(forbidden_chars, '_', filename)

    return sanitized


def token_distance(tokenizer, text, word1, word2):
    if word1 is None or word2 is None:
        return None

    if not (word1 in text and word2 in text):
        return None

    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    word1_tokens = tokenizer.tokenize(word1)
    word2_tokens = tokenizer.tokenize(word2)

    # Find the indices of the token spans corresponding to the words
    spans_word1 = [(i, i + len(word1_tokens)) for i in range(len(tokens) - len(word1_tokens) + 1) if
                   tokens[i:i + len(word1_tokens)] == word1_tokens]
    spans_word2 = [(i, i + len(word2_tokens)) for i in range(len(tokens) - len(word2_tokens) + 1) if
                   tokens[i:i + len(word2_tokens)] == word2_tokens]

    # Calculate the distance between all pairs of token spans
    distances = [abs(span1[0] - span2[0]) for span1 in spans_word1 for span2 in spans_word2]

    return min(distances) - len(word1_tokens) if distances else -1


def find_outliers(data_dict, threshold=3, top_k=None):
    # Convert the dictionary to a pandas Series
    data_series = pd.Series(data_dict)

    # Calculate the mean and standard deviation
    mean = np.mean(data_series)
    std = np.std(data_series)

    # Calculate the Z-scores
    z_scores = (data_series - mean) / std

    # Find entries with Z-scores greater than threshold (wrt absolute value)
    outliers = data_series[np.abs(z_scores) > threshold]

    # Sort outliers based on absolute z-score values
    outliers_sorted = outliers.iloc[np.argsort(np.abs(z_scores[outliers.index]))[::-1]]

    # Select top_k outliers based on highest absolute z-score values
    if top_k is not None and top_k > 0:
        outliers_sorted = outliers_sorted.head(top_k)

    return outliers_sorted

def evaluate_he_she(model, tokenizer, masked_text):
    """
    Evaluate the model on masked language modeling (MLM) task. Specifically, determine the probabilities of the tokens
    he and she in the masked text.

    Args:
    - model: The BERT model (BertForMaskedLM).
    - tokenizer: The BERT tokenizer.
    - masked_text: The text with a masked token (e.g., "The capital of France is [MASK].").
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
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Pass the inputs through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits and softmax to get probabilities
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]
    probabilities = torch.softmax(mask_token_logits, dim=-1)

    # Get the token IDs for "he" and "she"
    raw_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    he_token_id = raw_tokenizer.convert_tokens_to_ids("he")
    she_token_id = raw_tokenizer.convert_tokens_to_ids("she")

    # Get the probabilities for "he" and "she"
    shape = probabilities.shape

    if shape[0] == 0:
        result = {
            'he': [],
            'she': [],
            'most_likely_token': []
        }
    else:
        if is_generative:
            probabilities = probabilities.unsqueeze(0)

        he_probability = probabilities[:, he_token_id].tolist()
        she_probability = probabilities[:, she_token_id].tolist()

        # Determine the most likely token
        most_likely_token_id = torch.argmax(probabilities, dim=1).tolist()

        # Determine which token ("he" or "she") was the most likely
        most_likely_token = [tokenizer.decode(id) for id in most_likely_token_id]

        if shape[0] == 1:
            he_probability = he_probability[0]
            she_probability = she_probability[0]
            most_likely_token = most_likely_token[0]

        # Prepare the result dictionary
        result = {
            "he": he_probability,
            "she": she_probability,
            "most_likely_token": most_likely_token
        }

    return result


import fnmatch


def list_depth_1_entries(folder_path):
    entries = []

    # Top-level files and folders
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            entries.append(full_path)
        elif os.path.isdir(full_path):
            # Add files in subfolder (depth 1)
            for sub_entry in os.listdir(full_path):
                sub_path = os.path.join(full_path, sub_entry)
                if os.path.isfile(sub_path):
                    entries.append(sub_path)

    return entries

def get_files_and_folders_with_prefix(folder_path_prefix, suffix='', only_folder=False, only_files=False, **kwargs):
    assert not (only_folder and only_files), "only_folder and only_files cannot both be True"

    # Split the input into folder path and prefix
    folder_path, prefix = os.path.split(folder_path_prefix)

    # Get a list of all entries (files and folders) in the directory
    all_entries = list_depth_1_entries(folder_path)

    # Filter the list of entries to include only those that start with the prefix and suffix
    matching_entries = fnmatch.filter(all_entries, f"{prefix}*{suffix}")

    # Filter entries based on the only_folder flag
    if only_folder:
        matching_entries = [entry for entry in matching_entries if os.path.isdir(entry)]
    elif only_files:
        matching_entries = [entry for entry in matching_entries if os.path.isfile(entry)]

    for key, value in kwargs.items():
        key = f'{key[:3]}_{value}'
        matching_entries = [entry for entry in matching_entries if key in entry]

    return matching_entries

import hashlib
import torch

def hash_model_weights(model):

    if hasattr(model, 'hash'):
        model_hash = model.hash()
    else:
        # Create a BytesIO buffer to store the model's state_dict
        buffer = io.BytesIO()
        # Save the state_dict to the buffer
        torch.save(model.state_dict(), buffer)
        # Get the byte data from the buffer
        model_bytes = buffer.getvalue()
        # Create a SHA-256 hash
        model_hash = hashlib.sha256(model_bytes).hexdigest()
    return model_hash


# hashes a string deterministically across multiple program runs (in contrast to Python's built-in hash function)
def hash_it(x):
    """
    Returns a deterministic hash for any hashable object.

    Parameters:
    x (hashable): The object to hash.

    Returns:
    str: The SHA-256 hash of the object.
    """
    # Serialize the object to a byte stream using pickle
    byte_stream = pickle.dumps(x)

    # Compute the SHA-256 hash of the byte stream
    hash_object = hashlib.sha256(byte_stream)
    hash_hex = hash_object.hexdigest()

    return hash_hex


# Recursive function to convert tuple keys to strings
def convert_tuple_keys_to_strings(d):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            # Convert key to string if it's a tuple
            new_key = str(k) if isinstance(k, tuple) else k
            # Recursively apply to the value
            new_dict[new_key] = convert_tuple_keys_to_strings(v)
        return new_dict
    elif isinstance(d, list):
        # Apply conversion to each item if it's a list
        return [convert_tuple_keys_to_strings(item) for item in d]
    else:
        return d  # Return the item as-is if it's neither dict nor list

# Recursive function to convert string keys back to tuples
def convert_string_keys_to_tuples(d):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            # Convert key back to tuple if it looks like a tuple
            new_key = ast.literal_eval(k) if isinstance(k, str) and k.startswith('(') and k.endswith(')') else k
            # Recursively apply to the value
            new_dict[new_key] = convert_string_keys_to_tuples(v)
        return new_dict
    elif isinstance(d, list):
        # Apply conversion to each item if it's a list
        return [convert_string_keys_to_tuples(item) for item in d]
    else:
        return d  # Return the item as-is if it's neither dict nor list