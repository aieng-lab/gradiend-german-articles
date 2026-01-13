import copy
import re
import warnings
from collections import defaultdict

import numpy as np

import torch
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
import json
import os
from gradiend.util import hash_it, convert_tuple_keys_recursively

HF_TOKEN = os.getenv('HF_TOKEN')


class AutoModelForLM(nn.Module):

    @classmethod
    def from_pretrained(self, name_or_path, torch_dtype=torch.float32):
        try:
            model = AutoModelForMaskedLM.from_pretrained(name_or_path, trust_remote_code=True)
        except Exception:
            config_file = os.path.join(name_or_path, 'config_mlm_head.json')

            if os.path.exists(config_file):
                from gradiend.training.decoder_only_mlm.model import DecoderModelWithMLMHead
                return DecoderModelWithMLMHead.from_pretrained(name_or_path, torch_dtype=torch_dtype)
            elif 'llama' in name_or_path.lower():
                return AutoModelForCausalLM.from_pretrained(name_or_path,
                                                            torch_dtype=torch_dtype,
                                                            token=HF_TOKEN,
                                                            )
            model = AutoModelForCausalLM.from_pretrained(name_or_path)

        # set all requires_grad to True
        for param in model.parameters():
            param.requires_grad = True
        return model

class InstructTokenizerWrapper:
    system_prompt_mlm = """
    You are a language model that fills in masked words. In the following sentence, all [MASK] tokens refer to the same word. 
    Your task is to predict the missing word and return only that word — no explanation, no formatting, nothing else.
    """

    system_prompt = """
    You are a language model that completes sentences. Predict the next word that naturally follows the given text. 
    Return only that word — no punctuation, no quotes, and no explanations.
    """

    system_prompt_name = """
You are a language model trained to predict first names. In the following text, [NAME] represents a placeholder for a 
first name. Your task is to predict the most likely name that fits the context. Return only the predicted name — no 
punctuation, no quotation marks, and no explanations.
    """

    def __init__(self, tokenizer, user_prompt_header="user", assistant_prompt_header="assistant"):
        self.tokenizer = tokenizer
        self.user_prompt_header = user_prompt_header
        self.assistant_prompt_header = assistant_prompt_header

        # You can change these markers depending on the model
        self.BEGIN = "<|begin_of_text|>"
        self.START = "<|start_header_id|>"
        self.END = "<|end_header_id|>"
        self.EOT = "<|eot_id|>"

    def _wrap_prompt(self, user_text):
        if isinstance(user_text, str):
            user_texts = [user_text]
        elif isinstance(user_text, list):
            user_texts = user_text
        else:
            raise TypeError("user_text must be a string or a list of strings")

        prompts = []

        for text in user_texts:
            parts = [self.BEGIN]

            # Optional: add a system prompt first
            if self.system_prompt:
                parts.append(f"{self.START}system{self.END}\n{self.system_prompt}\n{self.EOT}")

            # Add user prompt
            parts.append(f"{self.START}{self.user_prompt_header}{self.END}\n{text}\n{self.EOT}")

            # Indicate the assistant is expected to reply
            parts.append(f"{self.START}{self.assistant_prompt_header}{self.END}\n")

            prompts.append(''.join(parts))

        return prompts if len(prompts) > 1 else prompts[0]

    def __call__(self, text, **kwargs):
        """
        Fully mimic Hugging Face tokenizer call: return dict with 'input_ids', 'attention_mask', etc.
        """
        if 'add_special_tokens' in kwargs and not kwargs['add_special_tokens']:
            # If add_special_tokens is False, we don't need to wrap the prompt
            wrapped = text
        else:
            wrapped = self._wrap_prompt(text)

        # our implementation adds special tokens by wrapping the prompt
        kwargs['add_special_tokens'] = False

        return self.tokenizer(wrapped, **kwargs)

    def tokenize(self, text, **kwargs):
        wrapped = self._wrap_prompt(text)
        return self.tokenizer.tokenize(wrapped, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def __getattr__(self, name):
        # Fallback to base tokenizer for any other attributes/methods
        return getattr(self.tokenizer, name)

    def __setattr__(self, key, value):
        if key in ['tokenizer', 'system_prompt']:
            super().__setattr__(key, value)
        else:
            setattr(self.tokenizer, key, value)

class AutoTokenizerForLM(AutoTokenizer):
    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(name, token=HF_TOKEN, *args, **kwargs)

        if "instruct" in name.lower():
            return InstructTokenizerWrapper(tokenizer)
        else:
            return tokenizer


def freeze_layers_until_target(model, *target_layer_names):
    assert len(target_layer_names) > 0, 'At least one target layer name must be provided'

    # iterate through the parameters from the model, starting at the input layer
    # we stop iterating as soon as we hit the first target layer
    for name, param in model.named_parameters():
        if name in target_layer_names:
            return

        param.requires_grad = False



def get_activation(activation: str, encoder=False):
    if activation == 'relu':
        activation_fnc = nn.ReLU(inplace=True)
    elif activation == 'leakyrelu':
        activation_fnc = nn.LeakyReLU(inplace=True)
    elif activation == 'tanh':
        activation_fnc = nn.Tanh()
    elif activation == 'smht':
        activation_fnc = nn.Hardtanh()
    elif activation == 'elu':
        activation_fnc = nn.ELU()
    elif activation == 'gelu':
        activation_fnc = nn.GELU()
    elif activation == 'sigmoid':
        activation_fnc = nn.Sigmoid()
    elif activation == 'id':
        if encoder:
            activation_fnc = nn.LayerNorm(1)
        else:
            activation_fnc = nn.Identity()
    else:
        raise ValueError('Unsupported activation function:', activation)
    return activation_fnc

class LargeLinear(nn.Module):
    """
        A linear layer that handles both standard and very large input feature sizes
        by using chunked computation when the input dimension exceeds the limit
        for standard CUDA BLAS GEMM operations (approximately 2.14 billion).

        Internally, it uses a standard `torch.nn.Linear` layer for parameter
        management and efficient computation for smaller inputs.

Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        in_chunk_size (int, optional): The size of the chunks to process the
            input dimension in when it exceeds the standard limit.
            Default: 2,000,000,000.
        out_chunk_size (int, optional): The size of the chunks to process the
            output dimension in when it exceeds the standard limit.
            Default: 2,000,000,000.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 in_chunk_size=2000000000,
                 out_chunk_size=2000000000,
                 dtype=torch.float32,
                 device=None,
                 ):
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.in_chunk_size = in_chunk_size
        self.out_chunk_size = out_chunk_size
        print('Dtype', dtype)
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype, device=device)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, input):
        if input.device != self.linear.weight.device:
            input = input.to(self.linear.weight.device)

        input_size = input.size(-1)
        output_size = self.out_features
        max_size = np.iinfo(np.int32).max
        if input_size <= max_size and output_size <= max_size:  # Standard computation if within limit
            return self.linear(input)
        elif output_size > max_size:
            # Chunking along the output dimension
            num_out_chunks = (output_size + self.out_chunk_size - 1) // self.out_chunk_size

            # determine output shape
            output_shape = list(input.shape[:-1]) + [output_size]
            output = torch.zeros(*output_shape, device=input.device, dtype=input.dtype)

            for i in range(num_out_chunks):
                out_start = i * self.out_chunk_size
                out_end = min((i + 1) * self.out_chunk_size, output_size)
                weight_chunk = self.linear.weight[out_start:out_end, :].to(input.device)
                bias_chunk = self.linear.bias[out_start:out_end] if self.linear.bias is not None else None

                if input_size > max_size:
                    # Also chunk along the input dimension
                    # todo test this if-branch!
                    num_in_chunks = (input_size + self.in_chunk_size - 1) // self.in_chunk_size
                    intermediate_parts = []
                    for j in range(num_in_chunks):
                        in_start = j * self.in_chunk_size
                        in_end = min((j + 1) * self.in_chunk_size, input_size)
                        input_chunk = input[..., in_start:in_end]
                        weight_in_chunk = weight_chunk[:, in_start:in_end]
                        output_in_chunk = torch.matmul(input_chunk.unsqueeze(-2),
                                                       weight_in_chunk.T.unsqueeze(-1)).squeeze(-1)
                        intermediate_parts.append(output_in_chunk)
                    output_part = torch.sum(torch.stack(intermediate_parts, dim=-1), dim=-1)
                else:
                    output_part = torch.matmul(input, weight_chunk.T).squeeze(-1)


                if bias_chunk is not None:
                    output_part += bias_chunk.to(input.device)

                output[..., out_start:out_end] = output_part

            return output
        elif input_size > max_size:
            # Chunked computation
            num_chunks = (input_size + self.in_chunk_size - 1) // self.in_chunk_size
            outputs = []
            for i in range(num_chunks):
                start = i * self.in_chunk_size
                end = min((i + 1) * self.in_chunk_size, input_size)
                input_chunk = input[..., start:end]
                weight_chunk = self.linear.weight[:, start:end].to(input.device)
                output_chunk = F.linear(input_chunk, weight_chunk, None) # Bias is added once at the end
                outputs.append(output_chunk)

            output = torch.sum(torch.stack(outputs, dim=-1), dim=-1)
            bias = self.linear.bias
            if bias is not None:
                output += bias.to(input.device)
            return output
        else:
            raise ValueError()

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value)
        self.linear.weight = value  # Avoid re-registering

    @property
    def bias(self):
        return self.linear.bias

    @bias.setter
    def bias(self, value):
        if value is not None and not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value)
        self.linear.bias = value  # Avoid re-registering


def gradiend_dir(load_directory):
    if not os.path.isdir(load_directory):
        raise FileNotFoundError(f"{load_directory} is not a directory")

    model_path = os.path.join(load_directory, 'pytorch_model.bin')
    if os.path.exists(model_path):
        return load_directory

    # this might be a folder with multiple GRADIENDs from an experimental training with different seeds
    # we pick the one with largest metrics.json value
    candidate_dirs = [os.path.join(load_directory, d) for d in os.listdir(load_directory)
                      if os.path.isdir(os.path.join(load_directory, d))]
    candidate_metrics = {}
    for cdir in candidate_dirs:
        metrics_file_path = os.path.join(cdir, 'metrics.json')
        if os.path.exists(metrics_file_path):
            with open(metrics_file_path, 'r') as f:
                metrics = json.load(f)
            if isinstance(metrics, (int, float)):
                candidate_metrics[cdir] = abs(metrics)
    if not candidate_metrics:
        raise FileNotFoundError(
            f"No model found in {load_directory} and no candidate subdirectories with metrics.json")

    best_dir = max(candidate_metrics, key=candidate_metrics.get)
    return best_dir



class GradiendModel(nn.Module):
    def __init__(self, input_dim,
                 latent_dim,
                 layers,
                 activation='tanh',
                 bias_decoder=True,
                 decoder_factor=1.0,
                 activation_decoder='id',
                 torch_dtype=torch.float32,
                 device=None,
                 device_encoder=None,
                 device_decoder=None,
                 **kwargs):
        super(GradiendModel, self).__init__()
        self.device_encoder = device_encoder or device or torch.device('cuda')
        self.device_decoder = device_decoder or device or torch.device('cuda')

        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)

        if self.input_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("Input and latent dimensions must be positive integers.")

        self.layers = layers
        self.activation = activation.lower()
        self.bias_decoder = bias_decoder
        self.torch_dtype = torch_dtype
        self.kwargs = kwargs

        activation_fnc = get_activation(self.activation, encoder=True)

        if activation_decoder:
            activation_fnc_decoder = get_activation(activation_decoder)
            self.activation_decoder = activation_decoder
        else:
            activation_fnc_decoder = get_activation(self.activation, encoder=False)
            self.activation_decoder = self.activation

        self.encoder = nn.Sequential(
            LargeLinear(input_dim, latent_dim, dtype=torch_dtype, device=self.device_encoder),
            activation_fnc
        )
        self.decoder = nn.Sequential(
            LargeLinear(latent_dim, input_dim, bias=bias_decoder, dtype=torch_dtype, device=self.device_decoder),
            activation_fnc_decoder
        )

        # Initialize the decoder weights with the same distribution as the encoder weights (up to decoder_factor)
        x = self.encoder[0].weight.max().item() * decoder_factor
        nn.init.uniform_(self.decoder[0].weight, -x, x)

        if bias_decoder:
            nn.init.uniform_(self.decoder[0].bias, -x, x)

        self.avg_gradient_norm = 0.0
        self.ctr = 0

    @property
    def decoder_norm(self):
        return torch.norm(self.decoder[0].weight, p=2).item()

    @property
    def encoder_norm(self):
        return torch.norm(self.encoder[0].weight, p=2).item()

    @property
    def layers_hash(self):
        if isinstance(self.layers, dict):
            layers_keys_hash = hash_it(list(self.layers.keys()))
            sparse_layers = torch.concat([self.layers[k].flatten() for k in self.layers], dim=0).cpu().to_sparse()
            layers_indices = sparse_layers.indices().cpu().numpy()
            layers_values = sparse_layers.values().cpu().numpy()
            layers_hash = hash_it((layers_keys_hash, layers_indices, layers_values))
        else:
            layers_hash = hash_it(self.layers)
        return layers_hash

    def extract_gradients_(self, bert, return_dict=False, ):
        layer_map = {k: v for k, v in bert.named_parameters()}
        if isinstance(self.layers, dict):
            if return_dict:
                return {k: v.grad.detach() * self.layers[k] if v.grad is not None else torch.zeros_like(v) for k, v in layer_map.items() if k in self.layers}
            else:
                layer_map = {k: v.grad[self.layers[k]] for k, v in layer_map.items() if k in self.layers}
            return torch.concat([layer_map[layer].flatten().detach() for layer in self.layers])
        elif return_dict:
            return {layer: layer_map[layer].grad.detach() for layer in self.layers}

        return torch.concat([layer_map[layer].grad.flatten().detach() for layer in self.layers])

    def extract_gradients(self, bert, return_dict=False):
        layer_map = {k: v for k, v in bert.named_parameters()}
        if isinstance(self.layers, dict):
            if return_dict:
                return {
                    k: (v.grad.detach().clone() * self.layers[k]
                        if v.grad is not None else torch.zeros_like(v))
                    for k, v in layer_map.items() if k in self.layers
                }
            else:
                layer_map = {
                    k: v.grad[self.layers[k]].detach().clone()
                    for k, v in layer_map.items() if k in self.layers
                }
                return torch.concat([layer_map[layer].flatten() for layer in self.layers])
        elif return_dict:
            return {layer: layer_map[layer].grad.detach().clone()
                    for layer in self.layers}

        return torch.concat([layer_map[layer].grad.flatten().detach().clone()
                             for layer in self.layers])

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, x, return_encoded=False):
        orig_shapes = {}

        if hasattr(x, 'named_parameters'):
            grads = []
            layer_map = {k: v for k, v in x.named_parameters()}
            if isinstance(self.layers, dict):
                # New behavior: process only masked elements
                for layer, mask in self.layers.items():
                    param = layer_map[layer]
                    grad = param.grad.flatten()
                    selected_grad = grad[mask.flatten()]  # Extract relevant elements
                    grads.append(selected_grad)
                    orig_shapes[layer] = (param.shape, mask)  # Store shape & mask
            else:
                for layer in self.layers:
                    param = layer_map[layer]
                    grad = param.grad.flatten()
                    grads.append(grad)
                    orig_shapes[layer] = param.shape
            x = torch.concat(grads)
        elif isinstance(x, dict):
            grads = []
            if isinstance(self.layers, dict):
                for layer, mask in self.layers.items():
                    param = x[layer]
                    grad = param.flatten()
                    selected_grad = grad[mask.flatten()]  # Extract relevant elements
                    grads.append(selected_grad)
                    orig_shapes[layer] = (param.shape, mask)  # Store shape & mask
            else:
                for layer in self.layers:
                    param = x[layer]
                    grad = param.flatten()
                    grads.append(grad)
                    orig_shapes[layer] = param.shape
            x = torch.concat(grads)

        encoded = self.encoder(x)
        if encoded.device != self.device_decoder:
            encoded = encoded.to(self.device_decoder)
        decoded = self.decoder(encoded)

        grad_norm = torch.norm(x, p=2).item()
        self.avg_gradient_norm = (self.avg_gradient_norm * self.ctr + grad_norm) / (self.ctr + 1)
        self.ctr += 1

        if orig_shapes:
            decoded_params = {}
            start_idx = 0
            for layer, shape_info in orig_shapes.items():
                if isinstance(shape_info, tuple):
                    shape, mask = shape_info  # Extract shape and mask
                    num_elements = mask.sum().item()

                    # Reconstruct full tensor, placing decoded values only where mask is True
                    reconstructed = torch.zeros_like(mask, dtype=decoded.dtype)
                    reconstructed[mask] = decoded[start_idx:start_idx + num_elements]
                    decoded_params[layer] = reconstructed.reshape(shape)
                else:
                    shape = shape_info
                    num_elements = shape.numel()
                    decoded_params[layer] = decoded[start_idx:start_idx + num_elements].reshape(shape)

            decoded = decoded_params

        if return_encoded:
            return decoded, encoded
        return decoded

    def forward_encoder(self, x):
        if hasattr(x, 'named_parameters'): # todo remove?
            grads = []
            layer_map = {k: v for k, v in x.named_parameters()}
            for layer in self.layers:
                param = layer_map[layer]
                grad = param.grad.flatten()
                grads.append(grad)

            x = torch.stack(grads)

        encoded = self.encoder(x)
        print('Encoded', encoded)
        return encoded

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        config_path = os.path.join(save_directory, 'config.json')
        layers_path = os.path.join(save_directory, 'layers.pth')

        # Save model state dictionary
        torch.save(self.state_dict(), model_path)

        self.kwargs.update(kwargs)

        # Save sparse tensor layers separately
        if isinstance(self.layers, dict):
            torch.save(self.layers, layers_path)

        # Save model configuration
        config = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'layers': list(self.layers),
            'activation': self.activation,
            'activation_decoder': self.activation_decoder,
            'bias_decoder': self.bias_decoder,
            **self._serialize_kwargs(),
        }
        if isinstance(self.layers, dict):
            config['layers_path'] = 'layers.pth'

        with open(config_path, 'w') as f:
            json.dump(config, f)



    def _serialize_kwargs(self):
        kwargs = self.kwargs.copy()
        if 'config' in kwargs['training']:
            training_kwargs = kwargs['training']['config'].copy()
        else:
            # todo remove depracated use!
            training_kwargs = kwargs['training']

        if isinstance(training_kwargs['layers'], dict):
            training_kwargs['layers'] = list(training_kwargs['layers'].keys())
            training_kwargs['layers_path'] = 'layers.pth'


        if 'config' in kwargs['training']:
            kwargs['training']['config'] = training_kwargs

        kwargs = convert_tuple_keys_recursively(kwargs)

        return kwargs

    @classmethod
    def _load_legacy_state_dict(cls, state_dict):
        warnings.warn(
            "You are using a legacy checkpoint format. Please update your model checkpoints. "
            "This fallback support will be removed in future versions.",
            DeprecationWarning
        )

        # Mapping from old keys to new keys
        key_mapping = {
            "encoder.0.weight": "encoder.0.linear.weight",
            "encoder.0.bias": "encoder.0.linear.bias",
            "decoder.0.weight": "decoder.0.linear.weight",
            "decoder.0.bias": "decoder.0.linear.bias",
        }

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = key_mapping.get(k, k)
            new_state_dict[new_key] = v

        return new_state_dict

    @classmethod
    def from_pretrained(cls, load_directory, device_encoder=None, device_decoder=None, device=None):
        model_path = os.path.join(load_directory, 'pytorch_model.bin')

        if not os.path.exists(model_path):
            best_dir = gradiend_dir(load_directory)
            print(f"Loading model from best candidate directory: {best_dir}")
            model_path = os.path.join(best_dir, 'pytorch_model.bin')
            load_directory = best_dir

        config_path = os.path.join(load_directory, 'config.json')

        # Load model configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        # check if the file actually belongs to a GRADIEND model
        if 'input_dim' not in config or 'latent_dim' not in config or 'layers' not in config:
            raise FileNotFoundError(f"The directory {load_directory} does not contain a valid GRADIEND model configuration.")

        device_encoder = device_encoder or device
        device_decoder = device_decoder or device

        if 'llama' in load_directory.lower() and device_encoder is None and device_decoder is None:
            # check that two GPUs are available
            if torch.cuda.device_count() < 2:
                raise RuntimeError("Two GPUs are required for GRADIEND Llama models.")

            device_encoder = torch.device("cuda:0")
            device_decoder = torch.device("cuda:1")

        # Instantiate the model
        model = cls(**config, device_encoder=device_encoder, device_decoder=device_decoder)

        # Load model state dictionary
        try:
            state_dict = torch.load(model_path, map_location=device_decoder, weights_only=True)
        except Exception as e:
            raise FileNotFoundError(f"Could not load model from {model_path}. Please ensure the file exists and is accessible.", e)

        # Check if the model is a legacy checkpoint
        if 'encoder.0.weight' in state_dict and 'decoder.0.weight' in state_dict:
            state_dict = cls._load_legacy_state_dict(state_dict)

        model.load_state_dict(state_dict)

        model.name_or_path = load_directory

        if 'layers_path' in config:
            layers_path = os.path.join(load_directory, config['layers_path'])
            # Load sparse layers
            try:
                model.layers = torch.load(layers_path)
            except FileNotFoundError:
                print(f"Warning: {layers_path} not found. Using all layers by default. This will be deprecated soon. Please do only specify layers_path in config if a layers_path exists")

        return model

    @property
    def grad_iterations(self):
        return self.kwargs.get('grad_iterations', 1)


    def plot(self, fig=None, bins=50, n=None):
        # Initialize lists to store weights and biases
        encoder_weights = []
        decoder_weights = []
        decoder_bias = []

        # Extract weights and biases from encoder and decoder
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                encoder_weights.append(param.flatten().detach().cpu().numpy())
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                decoder_weights.append(param.flatten().detach().cpu().numpy())
            elif 'bias' in name:
                decoder_bias.append(param.flatten().detach().cpu().numpy())

        # Convert lists to numpy arrays
        encoder_weights = np.concatenate(encoder_weights)
        decoder_weights = np.concatenate(decoder_weights)
        decoder_bias = np.concatenate(decoder_bias)

        # Compute bin edges and aggregate values
        encoder_bin_means, encoder_bin_edges, _ = binned_statistic(
            np.arange(len(encoder_weights)), encoder_weights, statistic='mean', bins=bins)
        decoder_bin_means, decoder_bin_edges, _ = binned_statistic(
            np.arange(len(encoder_weights), len(encoder_weights) + len(decoder_weights)),
            decoder_weights, statistic='mean', bins=bins)

        if self.bias_decoder:
            decoder_bias_bin_means, decoder_bias_bin_edges, _ = binned_statistic(
                np.arange(len(encoder_weights) + len(decoder_weights),
                          len(encoder_weights) + len(decoder_weights) + len(decoder_bias)),
                decoder_bias, statistic='mean', bins=bins)

        # Plotting
        if fig is None:
            fig = plt.figure(figsize=(12, 6))
        else:
            fig.clear()

        # Encoder weights (blue)
        plt.scatter(encoder_bin_edges[:-1], encoder_bin_means, color='blue', label='Encoder Weights')

        # Decoder weights (green)
        plt.scatter(decoder_bin_edges[:-1], decoder_bin_means, color='green', label='Decoder Weights')

        plt.title(f'Weights of Encoder and Decoder (Aggregated) {n if n else ""}')
        plt.xlabel('Parameter Index')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return

    @property
    def source(self):
        if 'config' in self.kwargs['training']:
            source = self.kwargs['training']['config']['source']
        else:
            # todo deprecate
            source = self.kwargs['training'].get('source', 'factual')

        if source == 'gradient':
            source = 'factual'
        elif source == 'inv_gradient':
            source = 'counterfactual'

        return source


    def __add__(self, other):
        if not isinstance(other, GradiendModel):
            raise ValueError("Can only add GradiendModel instances together.")

        # ensure both models have the same architecture
        if self.input_dim != other.input_dim or self.latent_dim != other.latent_dim:
            raise ValueError("Both GradiendModel instances must have the same input and latent dimensions.")

        new_model = copy.deepcopy(self)
        # Add encoder weights
        new_model.encoder[0].weight.data += other.encoder[0].weight.data
        if self.encoder[0].bias is not None and other.encoder[0].bias is not None:
            new_model.encoder[0].bias.data += other.encoder[0].bias.data
        # Add decoder weights
        new_model.decoder[0].weight.data += other.decoder[0].weight.data
        if self.decoder[0].bias is not None and other.decoder[0].bias is not None:
            new_model.decoder[0].bias.data += other.decoder[0].bias.data
        return new_model

    def __sub__(self, other):
        if not isinstance(other, GradiendModel):
            raise ValueError("Can only add GradiendModel instances together.")

        # ensure both models have the same architecture
        if self.input_dim != other.input_dim or self.latent_dim != other.latent_dim:
            raise ValueError("Both GradiendModel instances must have the same input and latent dimensions.")

        new_model = copy.deepcopy(self)
        # Add encoder weights
        new_model.encoder[0].weight.data -= other.encoder[0].weight.data
        if self.encoder[0].bias is not None and other.encoder[0].bias is not None:
            new_model.encoder[0].bias.data -= other.encoder[0].bias.data
        # Add decoder weights
        new_model.decoder[0].weight.data -= other.decoder[0].weight.data
        if self.decoder[0].bias is not None and other.decoder[0].bias is not None:
            new_model.decoder[0].bias.data -= other.decoder[0].bias.data
        return new_model


    def normalize(self):
        import copy
        new_model = copy.deepcopy(self)

        # Normalize encoder
        w = new_model.encoder[0].weight.data
        norm = w.norm(p=2)
        if norm > 0:
            new_model.encoder[0].weight.data = w / norm

        if new_model.encoder[0].bias is not None:
            b = new_model.encoder[0].bias.data
            norm_b = b.norm(p=2)
            if norm_b > 0:
                new_model.encoder[0].bias.data = b / norm_b

        # Normalize decoder
        w = new_model.decoder[0].weight.data
        norm = w.norm(p=2)
        if norm > 0:
            new_model.decoder[0].weight.data = w / norm

        if new_model.decoder[0].bias is not None:
            b = new_model.decoder[0].bias.data
            norm_b = b.norm(p=2)
            if norm_b > 0:
                new_model.decoder[0].bias.data = b / norm_b

        return new_model


    def get_top_k_weights(self,
                          *,
                          part='decoder',
                          top_k=100,
                          scope='neuron',  # 'weight' or 'neuron'
                          scope_direction='outgoing',
                          # 'outgoing' or 'incoming' (meaningful for 'neuron' or mapping weights->neurons)
                          return_sorted=True,
                          return_format='list',  # 'list' | 'by_param' | 'idx_to_param'
                          map_to_neuron=False  # only valid when scope == 'weight'
                          ):
        """
        Return top-k items from the specified part.

        Args:
            part: 'encoder' | 'decoder' | 'decoder_bias'
            top_k: number of top items to return
            scope: 'weight' (return top weight indices) or 'neuron' (return top neurons by aggregated weight importance)
            scope_direction: for neuron aggregation / mapping - 'outgoing' (output neurons / rows) or 'incoming' (input neurons / cols)
            return_format: 'list' (default) -> list of indices,
                           'by_param' -> dict param_name -> list,
                           'idx_to_param' -> dict index -> param_name
            map_to_neuron: if True and scope=='weight', also return mapping from weight index -> neuron index (ingoing/outgoing)

        Returns:
            Depending on return_format and map_to_neuron:
              - list of indices (default)
              - dicts as described above
              - if map_to_neuron True and return_format == 'list': returns tuple (indices_list, mapping_dict)
        """
        import numpy as np
        import torch

        assert scope in ['weight', 'neuron'], "scope must be 'weight' or 'neuron'"
        assert scope_direction in ['outgoing', 'incoming'], "scope_direction must be 'outgoing' or 'incoming'"
        assert return_format in ['list', 'by_param',
                                 'idx_to_param'], "return_format must be 'list','by_param' or 'idx_to_param'"

        if scope == 'neuron':
            raise ValueError('Use ModelWithGradiend.get_top_k_neurons for neuron-level importance.')


        # choose the correct parameter tensor
        if part == 'encoder':
            param = self.encoder[0].weight
            param_name = 'encoder.0.weight'
            bias = getattr(self.encoder[0], 'bias', None)
        elif part == 'decoder':
            param = self.decoder[0].weight
            param_name = 'decoder.0.weight'
            bias = getattr(self.decoder[0], 'bias', None)
        elif part == 'decoder_bias':
            # treat biases as a separate 1D tensor
            param = self.decoder[0].bias
            param_name = 'decoder.0.bias'
            bias = None
        else:
            raise ValueError("part must be 'encoder', 'decoder', or 'decoder_bias'")


        if param is None:
            # nothing to do
            if return_format == 'list':
                return []
            elif return_format == 'by_param':
                return {param_name: []}
            else:
                return {}

        w = param.data.squeeze().cpu()

        # compute importance according to scope
        if scope == 'weight':
            # flatten absolute weights
            importance = w.abs().flatten()
        else:  # scope == 'neuron'
            # For 2D weight matrix (out_features, in_features):
            if w.dim() == 2:
                if scope_direction == 'outgoing':
                    # each output neuron is a row -> sum abs across in_features
                    importance = w.abs().sum(dim=1)
                else:
                    # incoming: each input neuron is a column -> sum abs across out_features
                    importance = w.abs().sum(dim=0)
            elif w.dim() == 1:
                # bias: neuron-level already
                importance = w.abs()
            else:
                # fallback: flatten
                importance = w.abs().flatten()

        # clamp top_k
        k = min(int(top_k), int(importance.numel()))
        if k == 0:
            if return_format == 'list':
                return []
            elif return_format == 'by_param':
                return {param_name: []}
            else:
                return {}

        # topk (on CPU tensors)
        vals, top_idx = torch.topk(importance, k=k, largest=True, sorted=return_sorted)
        top_idx_list = top_idx.cpu().numpy().tolist()

        # optional mapping: weight index -> neuron index
        neuron_map = None
        if map_to_neuron and scope == 'weight':
            # Need the original weight shape to map flattened weight index -> (out_idx, in_idx)
            if w.dim() == 2:
                out_dim, in_dim = w.shape
                # unravel top indices to (out, in)
                idx_arr = np.array(top_idx.cpu().numpy(), dtype=np.int64)
                out_idx, in_idx = np.unravel_index(idx_arr, (out_dim, in_dim))
                if scope_direction == 'outgoing':
                    mapped = out_idx.tolist()
                else:
                    mapped = in_idx.tolist()
                neuron_map = dict(zip(top_idx_list, mapped))
            elif w.dim() == 1:
                # flattened weights is same as neurons for 1D (bias) -> map to that index
                neuron_map = {int(i): int(i) for i in top_idx_list}
            else:
                # unknown shape: return None mapping
                neuron_map = {int(i): None for i in top_idx_list}

        # assemble return according to requested format
        if return_format == 'list':
            if map_to_neuron and scope == 'weight':
                return top_idx_list, neuron_map
            return top_idx_list

        elif return_format == 'by_param':
            # currently we only support a single param (encoder.0.weight / decoder.0.weight / decoder.0.bias)
            if map_to_neuron and scope == 'weight':
                return {param_name: {'top_indices': top_idx_list, 'neuron_map': neuron_map}}
            return {param_name: top_idx_list}

        else:  # 'idx_to_param'
            # map each returned (flat) index -> param name
            if map_to_neuron and scope == 'weight':
                return {int(idx): {'param': param_name, 'neuron': neuron_map[int(idx)]} for idx in top_idx_list}
            return {int(idx): param_name for idx in top_idx_list}



    def _get_top_k_neurons(
            self,
            model: torch.nn.Module,
            weight_scores: torch.Tensor,  # flattened
            k: int,
            direction="outgoing",
    ):
        """
        Returns:
            weight_to_neuron: dict[int -> neuron_id]
            neuron_to_all_weights: dict[neuron_id -> list[int]]
        """

        assert direction in {"outgoing", "ingoing"}

        topk_indices = torch.topk(weight_scores.abs(), k).indices.tolist()
        topk_set = set(topk_indices)

        weight_to_neuron = {}
        neuron_to_all_weights = {}

        config = model.config
        running_idx = 0

        for name, param in model.named_parameters():
            if self.layers and name not in self.layers:
                continue


            if param.dim() < 2:
                running_idx += param.numel()
                continue

            rows, cols = param.shape
            numel = param.numel()
            param_range = range(running_idx, running_idx + numel)

            overlap = topk_set.intersection(param_range)
            if not overlap:
                running_idx += numel
                continue

            local_topk = [i - running_idx for i in overlap]
            lname = name.lower()

            # ============================================================
            # GPT-2: fused QKV (attn.c_attn)
            # ============================================================
            if "attn.c_attn" in lname:
                hidden = rows
                num_heads = config.n_head
                dk = hidden // num_heads

                cols_hit = {idx % cols for idx in local_topk}

                for c in cols_hit:
                    block = c // hidden  # 0=Q,1=K,2=V
                    inner = c % hidden
                    head = inner // dk
                    j = inner % dk
                    role = ["q", "k", "v"][block]

                    neuron = ("attention", name, role, head, j)

                    # top-k → neuron
                    for r in range(rows):
                        widx = running_idx + r * cols + c
                        if widx in overlap:
                            weight_to_neuron[widx] = neuron

                    # ALL weights
                    all_w = [
                        running_idx + r * cols + c
                        for r in range(rows)
                    ]
                    neuron_to_all_weights.setdefault(neuron, all_w)

            # ============================================================
            # LLaMA: q_proj / k_proj / v_proj (GQA-safe)
            # ============================================================
            elif any(x in lname for x in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj"
            ]):
                hidden = config.hidden_size
                num_heads = config.num_attention_heads
                dk = hidden // num_heads

                if "q_proj" in lname:
                    role = "q"
                elif "k_proj" in lname:
                    role = "k"
                else:
                    role = "v"

                cols_hit = {idx % cols for idx in local_topk}

                for c in cols_hit:
                    head = c // dk
                    j = c % dk
                    neuron = ("attention", name, role, head, j)

                    for r in range(rows):
                        widx = running_idx + r * cols + c
                        if widx in overlap:
                            weight_to_neuron[widx] = neuron

                    all_w = [
                        running_idx + r * cols + c
                        for r in range(rows)
                    ]
                    neuron_to_all_weights.setdefault(neuron, all_w)

            # ============================================================
            # BERT-style attention (query / key / value)
            # ============================================================
            elif any(x in lname for x in ["query", "key", "value"]):
                base = name.replace(".weight", "").replace(".bias", "")
                base = base.rsplit(".", 1)[0]

                num_heads = config.num_attention_heads
                dk = cols // num_heads

                if "query" in lname:
                    role = "q"
                elif "key" in lname:
                    role = "k"
                else:
                    role = "v"

                cols_hit = {idx % cols for idx in local_topk}

                for c in cols_hit:
                    head = c // dk
                    j = c % dk
                    neuron = ("attention", base, head, j)

                    for r in range(rows):
                        widx = running_idx + r * cols + c
                        if widx in overlap:
                            weight_to_neuron[widx] = neuron

                    all_w = [
                        running_idx + r * cols + c
                        for r in range(rows)
                    ]
                    neuron_to_all_weights.setdefault(neuron, all_w)

            # ============================================================
            # LLaMA MLP (SwiGLU: gate / up / down)
            # ============================================================
            elif any(x in lname for x in ["gate_proj", "up_proj", "down_proj"]):
                rows_hit = {idx // cols for idx in local_topk}
                for r in rows_hit:
                    neuron = ("mlp", name, r)

                    for c in range(cols):
                        widx = running_idx + r * cols + c
                        if widx in overlap:
                            weight_to_neuron[widx] = neuron

                    all_w = [
                        running_idx + r * cols + c
                        for c in range(cols)
                    ]
                    neuron_to_all_weights.setdefault(neuron, all_w)

            # ============================================================
            # Standard MLP (BERT / GPT-2)
            # ============================================================
            else:
                if direction == "outgoing":
                    rows_hit = {idx // cols for idx in local_topk}
                    for r in rows_hit:
                        neuron = ("mlp", name, r)

                        for c in range(cols):
                            widx = running_idx + r * cols + c
                            if widx in overlap:
                                weight_to_neuron[widx] = neuron

                        all_w = [
                            running_idx + r * cols + c
                            for c in range(cols)
                        ]
                        neuron_to_all_weights.setdefault(neuron, all_w)

                else:  # ingoing
                    cols_hit = {idx % cols for idx in local_topk}
                    for c in cols_hit:
                        neuron = ("mlp", name, c)

                        for r in range(rows):
                            widx = running_idx + r * cols + c
                            if widx in overlap:
                                weight_to_neuron[widx] = neuron

                        all_w = [
                            running_idx + r * cols + c
                            for r in range(rows)
                        ]
                        neuron_to_all_weights.setdefault(neuron, all_w)

            running_idx += numel

        return weight_to_neuron, neuron_to_all_weights


    def __len__(self):
        return self.encoder[0].in_features


def is_generative(model):
    return hasattr(model, 'lm_head') or 'llama' in model.__class__.__name__.lower()




class ModelWithGradiend(nn.Module):

    def __init__(self, base_model, gradiend, tokenizer, base_model_device=None, torch_dtype=None):
  
        super().__init__()
        self.base_model = base_model
        self.gradiend = gradiend
        self.tokenizer = tokenizer
        self.grad_iterations = gradiend.grad_iterations

        self.base_model_device = base_model_device or gradiend.encoder[0].linear.weight.device
        self.base_model.to(self.base_model_device)
        if torch_dtype:
            self.base_model = self.base_model.to(torch_dtype)
            self.gradiend = self.gradiend.to(torch_dtype)

        self.layer_map = {k: v for k, v in self.base_model.named_parameters()}

        self.is_instruction_model = isinstance(self.tokenizer, InstructTokenizerWrapper)
        self.is_generative = is_generative(self.base_model)

        if self.is_generative:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def name(self):
        return os.path.basename(self.gradiend.name_or_path)

    def create_gradients(self, text, label, return_dict=False, verbose=False):
       
        item = self.create_inputs(text, label)

        outputs = self.base_model(**item)


        if verbose:
            # print the predicted inputs/labels
            labels = item['labels']
            input_ids = item['input_ids']
            #label_idx = labels[labels != -100]
            if self.is_generative:
                logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
                predictions = logits.argmax(dim=-1)  # shape: [batch_size, seq_len]

                # Loop through batch
                for i in range(input_ids.size(0)):
                    label_mask = labels[i] != -100
                    label_positions = label_mask.nonzero(as_tuple=True)[0]

                    print(f"\n[Sample {i}]")
                    for pos in label_positions:
                        pred_id = predictions[i, pos-1].item()
                        true_id = labels[i, pos].item()

                        pred_token = self.tokenizer.decode([pred_id], skip_special_tokens=True)
                        true_token = self.tokenizer.decode([true_id], skip_special_tokens=True)

                        print(f"Position {pos.item():>2}: Predicted = '{pred_token}', True = '{true_token}'",
                              "✅" if pred_token.lower() == true_token.lower() else "❌")

        loss_base_model = outputs.loss

        self.base_model.zero_grad()
        loss_base_model.backward()

        gradients = self.gradiend.extract_gradients(self.base_model, return_dict=return_dict)
        return gradients

    # do i even need this if i change the name....
    def encode(self, text, label):
        gradients = self.create_gradients(text, label)
        # if isinstance(self.gradiend, CombinedEncoderDecoder): 
        #     encoded = self.gradiend.encode(gradients, shared=shared)
        # else:
        encoded = self.gradiend.encoder(gradients)
        return encoded



    def transform_weights_to_neurons(self):
        raise NotImplementedError()



    def modify_model(self, lr, feature_factor, part='decoder',  top_k=None, top_k_part=None, additive=True):
        from gradiend.combined_models.combined_gradiends import StackedGradiend, CombinedEncoderDecoder
        # returns a base_model model with enhanced weights based on the auto encoder, use the learning rate parameter to control the influence of the auto encoder
        top_k_part = top_k_part or part

        import copy
        enhanced_model = copy.deepcopy(self.base_model)

        if top_k == 0:
            return enhanced_model

        model_device = self.base_model.device
        layer_map = {k: v for k, v in enhanced_model.named_parameters()}
        if part == 'decoder':
            if isinstance(self.gradiend, StackedGradiend):
                enhancer1, enhancer2, enhancer3 = self.gradiend.modify_model_decode_v1(feature_factor)
            elif isinstance(self.gradiend, CombinedEncoderDecoder):
                enhancer = self.gradiend.modified_decode(x = feature_factor, method='sum')
            else:
                decoder_dtype = self.gradiend.decoder[0].weight.dtype
                enhancer = self.gradiend.decoder(torch.tensor([feature_factor], dtype=decoder_dtype, device=model_device))
        elif part == 'encoder':
            enhancer = self.gradiend.encoder[0].weight.flatten().to(model_device)
        else:
            raise ValueError('Unsupported part:', part)

        if top_k is not None and top_k < len(enhancer):
            mask = self.get_enhancer_mask(top_k, part=top_k_part)

            if mask.sum() == 0.0:
                return enhanced_model

            enhancer1[~mask] = 0.0

        idx = 0
        with torch.no_grad():
            if isinstance(self.gradiend.layers, dict):
                for layer, layer_mask in self.gradiend.layers.items():
                    number_of_elements = layer_mask.sum().item()  # Convert to Python integer if it's a tensor

                    # Extract the relevant elements from enhancer
                    update_values = enhancer[idx:idx + number_of_elements].to(model_device)

                    # Create an update tensor of the same shape as layer_mask, filling only True positions
                    update_tensor = torch.zeros_like(layer_mask, dtype=update_values.dtype)
                    update_tensor[layer_mask] = update_values  # Assign values only to True positions

                    # Apply update
                    if additive:
                        layer_map[layer] += lr * update_tensor
                    else: # todo
                        layer_map[layer] = lr * torch.ones_like(update_tensor)

                    # Increment index
                    idx += number_of_elements
            else:
                for layer in self.gradiend.layers:
                    shape = layer_map[layer].shape
                    number_of_elements = shape.numel()
                    if  isinstance(self.gradiend, StackedGradiend):
                        layer_chunk_1 = enhancer1[idx:idx + number_of_elements].to(model_device)
                        layer_chunk_2 = enhancer2[idx:idx + number_of_elements].to(model_device)
                        layer_chunk_3 = enhancer3[idx:idx + number_of_elements].to(model_device)
                        layer_chunk = torch.mean(torch.stack([layer_chunk_1, layer_chunk_2, layer_chunk_3]), dim=0)
                    else:   
                        layer_chunk = enhancer[idx:idx + number_of_elements].to(model_device)
                    if additive:
                        layer_map[layer] += lr * layer_chunk.reshape(shape)
                    else:

                        layer_map[layer] = lr * torch.ones_like(layer_chunk) # todo
                    idx += number_of_elements

        return enhanced_model

    def modify_model(self, lr, feature_factor, part="decoder", top_k=None, top_k_part=None, additive=True):
        import copy
        import torch
        from gradiend.combined_models.combined_gradiends import StackedGradiend, CombinedEncoderDecoder

        top_k_part = top_k_part or part
        enhanced_model = copy.deepcopy(self.base_model)

        if top_k == 0:
            return enhanced_model

        def _first_param_device_dtype(m):
            p = next(m.parameters(), None)
            if p is None:
                return torch.device("cpu"), torch.float32
            return p.device, p.dtype

        def _as_like_scalar(x, *, device, dtype):
            # robust for python floats/ints, numpy scalars, tensors
            return torch.as_tensor(x, device=device, dtype=dtype)

        def _cast_to_param(t, param, *, flatten=False):
            t = t.to(device=param.device, dtype=param.dtype, non_blocking=True)
            return t.flatten() if flatten else t

        # --- build enhancer on the enhancer's own device/dtype (NOT the base model's) ---
        if part == "decoder":
            if isinstance(self.gradiend, StackedGradiend):
                enhancer1, enhancer2, enhancer3 = self.gradiend.modify_model_decode_v1(feature_factor)
            elif isinstance(self.gradiend, CombinedEncoderDecoder):
                enhancer = self.gradiend.modified_decode(x=feature_factor, method="sum")
            else:
                dec = self.gradiend.decoder
                enh_dev, enh_dtype = _first_param_device_dtype(dec)
                ff = _as_like_scalar([feature_factor], device=enh_dev, dtype=enh_dtype)
                enhancer = dec(ff)
        elif part == "encoder":
            enc = self.gradiend.encoder
            enh_dev, enh_dtype = _first_param_device_dtype(enc)
            # keep on enhancer device for now; cast per-layer later
            enhancer = enc[0].weight.detach().flatten().to(device=enh_dev)
        else:
            raise ValueError(f"Unsupported part: {part}")

        # --- top-k masking (works for both stacked and single enhancer) ---
        if top_k is not None:
            if isinstance(self.gradiend, StackedGradiend):
                enh_len = len(enhancer1)
            else:
                enh_len = len(enhancer)

            if top_k < enh_len:
                mask = self.get_enhancer_mask(top_k, part=top_k_part)
                if not torch.is_tensor(mask):
                    mask = torch.as_tensor(mask)
                mask = mask.to(dtype=torch.bool)

                if mask.sum().item() == 0:
                    return enhanced_model

                if isinstance(self.gradiend, StackedGradiend):
                    # ensure mask sits on the right device for each tensor
                    enhancer1 = enhancer1.clone()
                    enhancer2 = enhancer2.clone()
                    enhancer3 = enhancer3.clone()
                    enhancer1[~mask.to(enhancer1.device)] = 0.0
                    enhancer2[~mask.to(enhancer2.device)] = 0.0
                    enhancer3[~mask.to(enhancer3.device)] = 0.0
                else:
                    enhancer = enhancer.clone()
                    enhancer[~mask.to(enhancer.device)] = 0.0

        # --- apply updates per-parameter on THAT parameter's device/dtype ---
        layer_map = dict(enhanced_model.named_parameters())
        idx = 0

        with torch.no_grad():
            if isinstance(self.gradiend.layers, dict):
                # layers: {param_name: boolean_mask_tensor_same_shape_as_param}
                for layer_name, layer_mask in self.gradiend.layers.items():
                    param = layer_map[layer_name]
                    m = layer_mask
                    if not torch.is_tensor(m):
                        m = torch.as_tensor(m)
                    m = m.to(device=param.device, dtype=torch.bool)

                    n = int(m.sum().item())
                    if isinstance(self.gradiend, StackedGradiend):
                        raise ValueError("Dict-style layers with StackedGradiend not supported in current code path.")
                    if idx + n > len(enhancer):
                        raise ValueError("Enhancer vector shorter than required parameter updates.")

                    upd_vals = _cast_to_param(enhancer[idx: idx + n], param, flatten=True)

                    # fill masked positions robustly regardless of mask shape
                    upd = torch.zeros_like(param, dtype=param.dtype, device=param.device)
                    upd_flat = upd.view(-1)
                    m_flat = m.view(-1)
                    upd_flat[m_flat] = upd_vals
                    upd = upd_flat.view_as(param)

                    if additive:
                        param.data.add_(upd, alpha=lr)
                    else:
                        param.data.copy_(torch.ones_like(param) * lr)

                    idx += n
            else:
                # layers: list of param_names; enhancer is flat chunk per param.numel()
                for layer_name in self.gradiend.layers:
                    param = layer_map[layer_name]
                    n = param.numel()

                    if isinstance(self.gradiend, StackedGradiend):
                        if idx + n > len(enhancer1):
                            raise ValueError("Enhancer vector shorter than required parameter updates.")
                        c1 = _cast_to_param(enhancer1[idx: idx + n], param, flatten=True)
                        c2 = _cast_to_param(enhancer2[idx: idx + n], param, flatten=True)
                        c3 = _cast_to_param(enhancer3[idx: idx + n], param, flatten=True)

                        # mean in fp32 for stability, then cast back to param dtype
                        chunk = torch.stack([c1.float(), c2.float(), c3.float()], dim=0).mean(dim=0).to(param.dtype)
                    else:
                        if idx + n > len(enhancer):
                            raise ValueError("Enhancer vector shorter than required parameter updates.")
                        chunk = _cast_to_param(enhancer[idx: idx + n], param, flatten=True)

                    chunk = chunk.view_as(param)

                    if additive:
                        param.data.add_(chunk, alpha=lr)
                    else:
                        param.data.copy_(torch.ones_like(param) * lr)

                    idx += n

        return enhanced_model

    def get_enhancer(self, part='decoder'):
        # we set parts of the enhancer to 0 that are not in the top k highest values (wrt absolute value)
        if part == 'decoder':
            abs_enhancer = self.gradiend.decoder[0].weight.flatten().abs()
        elif part == 'decoder-bias':
            abs_enhancer = self.gradiend.decoder[0].bias.abs()
        elif part == 'decoder-sum':
            abs_enhancer = (self.gradiend.decoder[0].weight.flatten() + self.gradiend.decoder[0].bias).abs()
        else:
            raise ValueError('Unsupported part:', part)

        return abs_enhancer


    def get_enhancer_mask(self, top_k, part='decoder'):
        gradiend_vector = self.gradiend.decoder[0].weight.flatten()

        if 0.0 < top_k <= 1.0:
            top_k = int(top_k * len(gradiend_vector))

        abs_enhancer = self.get_enhancer(part=part).cpu() # move to cpu to ensure deterministic behavior
        sorted_indices = torch.argsort(abs_enhancer, stable=True)  # Ensure stable order
        sorted_enhancer = abs_enhancer[sorted_indices]

        top_k_values, top_k_sorted_indices = sorted_enhancer.topk(top_k, sorted=False, largest=True)

        # Convert back to original indices
        top_k_indices = sorted_indices[top_k_sorted_indices]
        mask = torch.zeros_like(gradiend_vector, dtype=torch.bool)
        mask[top_k_indices] = True
        return mask

    def get_layer_mask(self, top_k, part='decoder'):
        enhancer_mask = self.get_enhancer_mask(top_k=top_k, part=part)
        all_layer_map = {k: v for k, v in self.base_model.named_parameters()}
        layer_map = {}

        idx = 0
        with torch.no_grad():
            if isinstance(self.gradiend.layers, dict):
                raise NotImplementedError()
                for layer, layer_mask in self.gradiend.layers.items():
                    shape = layer_map[layer].shape
                    number_of_elements = layer_mask.sum().item()  # Convert to Python integer if it's a tensor

                    # Extract the relevant elements from enhancer
                    #update_values = enhancer[idx:idx + number_of_elements]

                    # Create an update tensor of the same shape as layer_mask, filling only True positions
                    update_tensor = torch.zeros_like(layer_mask, dtype=update_values.dtype)
                    update_tensor[layer_mask] = update_values  # Assign values only to True positions

                    # Apply update
                    layer_map[layer] = lr * update_tensor

                    # Increment index
                    idx += number_of_elements
            else:
                for layer in self.gradiend.layers:
                    shape = all_layer_map[layer].shape
                    number_of_elements = shape.numel()
                    layer_map[layer] = enhancer_mask[idx:idx + number_of_elements].reshape(shape) #.to_sparse()
                    idx += number_of_elements

        return layer_map


    def mask_and_encode(self, text, ignore_tokens=False, return_masked_text=False, single_mask=True, shared=False):
        item = self.tokenizer(text, return_tensors="pt")
        item = {k: v.to(self.base_model_device) for k, v in item.items()}
        labels = item['input_ids'].clone()

        if self.is_generative:

            n = labels.shape[1]

            # left shift the labels by one
            labels = torch.cat([labels[:, 1:], torch.full_like(labels[:, :1], self.tokenizer.pad_token_id)], dim=1)

            # use random idx to predict in the 2nd half of the sequence to ensure enough context
            if ignore_tokens:
                # Create a mask for valid indices in the second half, excluding the ignore tokens
                valid_indices_mask = ~torch.isin(labels[0, n // 2:], torch.tensor(ignore_tokens, device=labels.device))

                # Get the indices of the valid tokens
                valid_indices = torch.nonzero(valid_indices_mask, as_tuple=False).squeeze()

                if valid_indices.numel() > 0:
                    # Randomly select an index from the valid tokens
                    random_idx = valid_indices[torch.randint(0, valid_indices.numel(), (1,))] + n // 2
                else:
                    # Handle case where no valid indices exist (fallback behavior, could raise an error or select any index)
                    raise ValueError('No valid indices found in the second half of the sequence', text)
            else:
                random_idx = torch.randint(n // 2, n, (1,))

            labels[:, :random_idx-1] = self.tokenizer.pad_token_id
            labels[:, random_idx:] = self.tokenizer.pad_token_id

            mask = labels != self.tokenizer.pad_token_id
        else:
            if single_mask:
                mask = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
                # randomly mask a single entry
                mask[0, torch.randint(0, labels.shape[1], (1,))] = True
            else:
                random_mask = torch.rand(labels.shape, dtype=torch.float, device=labels.device) < 0.15
                padding_mask = labels == self.tokenizer.pad_token_id

                if ignore_tokens:
                    exclude_mask = (labels.unsqueeze(-1) == torch.Tensor(ignore_tokens).to(labels.device)).any(dim=-1)
                else:
                    exclude_mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
                mask = random_mask & ~padding_mask & ~exclude_mask
            labels[~mask] = -100  # only compute loss on masked tokens

            item['input_ids'][mask] = self.tokenizer.mask_token_id

        item['labels'] = labels

        outputs = self.base_model(**item)
        loss_bert = outputs.loss

        self.base_model.zero_grad()
        if loss_bert is None:
            return None
        loss_bert.backward()

        gradients = self.gradiend.extract_gradients(self.base_model)

        if shared: 
            encoded = self.gradiend.encode(gradients, shared=False).detach().cpu().numpy()
        else:
            encoded = self.gradiend.encoder(gradients).to(torch.float32).detach().cpu().numpy()

        if return_masked_text:
            masked_str = self.tokenizer.decode(item['input_ids'].squeeze())
            masked_str = masked_str.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
            labels = [self.tokenizer.decode(id) for id in item['labels'][mask]]
            return encoded, masked_str, labels

        return encoded


    def create_inputs(self, masked_text, label):
        item = self.tokenizer(masked_text, return_tensors="pt")
        item = {k: v.to(self.base_model_device) for k, v in item.items()}
        if hasattr(self.tokenizer, 'tokenizer'):
            label_token_id = self.tokenizer.tokenizer(f' {label}', add_special_tokens=False)['input_ids']
        else:
            label_token_id = self.tokenizer(f'{label}', add_special_tokens=False)['input_ids']
        if not len(label_token_id) == 1:
            raise ValueError('Only a single label token is supported', label_token_id, label)
        label_token_id = label_token_id[0]

        if self.is_generative:
            item['labels'] = torch.full_like(item['input_ids'], -100)
            last_idx = (item['input_ids'].squeeze() != self.tokenizer.pad_token_id).nonzero()[-1].item()
            #last_idx -= int(self.is_instruction_model)
            item['labels'][:, last_idx] = label_token_id
        else:
            labels = item['input_ids'].clone()
            labels[labels != self.tokenizer.mask_token_id] = -100  # only compute loss on masked tokens
            labels[labels == self.tokenizer.mask_token_id] = label_token_id
            item['labels'] = labels
        return item

    def forward_pass(self, inputs, return_dict=False, lr=1e-4, verbose=False): # todo implement batched=True

        inputs = {k: v.to(self.base_model_device) for k, v in inputs.items()}

        grads = []

        if self.grad_iterations > 1:
            base_model = copy.deepcopy(self.base_model)
            for i in range(self.grad_iterations):
                outputs = base_model(**inputs)
                loss_bert = outputs.loss

                base_model.zero_grad()

                if loss_bert is None:
                    return None
                loss_bert.backward()
                gradients = self.gradiend.extract_gradients(base_model, return_dict=True)
                grads.append(gradients)

                if i < self.grad_iterations - 1:
                    # perform the training step
                    # Step 6: Update the model's weights

                    with torch.no_grad():
                        for name, param in base_model.named_parameters():
                            if param.grad is not None:
                                param.add_(-lr * param.grad)

                # save only the last gradient (for now, maybe add some advanced logic later)
                grads = [grads[-1]]

            if return_dict:
                return grads[-1]
            flatten_gradient = torch.concat(tuple(grad.flatten() for grad in grads[-1].values()))
            return flatten_gradient

        else:
            outputs = self.base_model(**inputs)

            if verbose:
                # todo check gpt
                # Get most likely next token from CausalLMOutputWithPast
                # Find the last valid (non -100) label index for each sequence
                labels = inputs['labels']
                last_valid_indices = (labels != -100).int().argmax(dim=1, keepdim=True)
                pad_id = self.tokenizer.pad_token_id
                mask = inputs['input_ids'] != pad_id  # shape: [B, T]
                # But this gives you the first non-pad; we want last
                last_non_pad_indices = (inputs['input_ids'] != pad_id).int().flip(dims=[1]).argmax(dim=1)
                last_non_pad_indices = inputs['input_ids'].size(1) - 2 - last_non_pad_indices #+ int(self.is_instruction_model)

                # Get logits at those positions
                batch_indices = torch.arange(labels.size(0), device=labels.device)
                selected_logits = outputs.logits[batch_indices, last_non_pad_indices,:]  # shape: (batch_size, vocab_size)

                # Get most likely next token
                next_token_ids = selected_logits.argmax(dim=-1)  # shape: (batch_size,)
                #print('Next Tokens', next_tokens)

            loss_bert = outputs.loss
            if loss_bert is None:
                return None

            self.base_model.zero_grad()
            loss_bert.backward()
            return self.gradiend.extract_gradients(self.base_model, return_dict=return_dict)


    @property
    def layers_hash(self):
        return self.gradiend.layers_hash

    def __len__(self):
        return len(self.gradiend)

    # invert the encoded value, i.e. encoded value * (-1), while keeping the decoders value
    def invert_encoding(self):
        with torch.no_grad():
            self.gradiend.encoder[0].weight.data *= -1
            self.gradiend.encoder[0].bias.data *= -1
            self.gradiend.decoder[0].weight.data *= -1

    def save_pretrained(self, save_directory, **kwargs):
        self.gradiend.save_pretrained(save_directory, bert=self.base_model.name_or_path, tokenizer=self.tokenizer.name_or_path, **kwargs)

    @classmethod
    def from_pretrained(cls, load_directory, ae=None, layers=None, latent_dim=1, torch_dtype=torch.float32, ensemble=False, device=None, device_decoder=None, **kwargs):
        layers = layers or []
        if len(layers) == 1 and isinstance(layers[0], list):
            layers = layers[0]

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_encoder = device
        device_decoder = device
        device_base_model = device
        if 'llama' in load_directory.lower() and device != torch.device("cpu"):
            # check that two GPUs are available
            cuda_count = torch.cuda.device_count()
            if cuda_count < 2:
                raise RuntimeError("Two GPUs are required for GRADIEND Llama models.")

            device_encoder = torch.device("cuda:0")
            device_decoder = torch.device("cuda:1")
            available_gpu_ram = torch.cuda.get_device_properties(device_encoder).total_memory // (1024 ** 3)
            available_gpu_ram_greater_than_100GB = available_gpu_ram >= 100
            device_base_model = device_encoder if (cuda_count == 2 or available_gpu_ram_greater_than_100GB) else torch.device("cuda:2")
            print(f'Using device_encoder: {device_encoder}, device_decoder: {device_decoder}, device_base_model: {device_base_model}')
            torch_dtype=torch.bfloat16

        if '1B' in load_directory.lower():
            torch_dtype = torch.bfloat16

        try:
            if not ensemble: 
                ae = GradiendModel.from_pretrained(load_directory, device_encoder=device_encoder, device_decoder=device_decoder)

            if layers and ae.layers != layers:
                raise ValueError(f'The provided layers {layers} do not match the layers in the model configuration {ae.layers}')
            else:
                layers = ae.layers

            if ensemble: 
                base_model_id = ae.kwargs['bert']
                tokenizer = ae.kwargs['tokenizer']
            else:
                base_model_id = ae.kwargs['base_model']
                tokenizer = ae.kwargs.get('tokenizer', base_model_id)

            base_model = AutoModelForLM.from_pretrained(base_model_id, torch_dtype=torch_dtype).to(device_base_model)
            tokenizer = AutoTokenizerForLM.from_pretrained(tokenizer)
        except FileNotFoundError:
            print('No model with auto encoder found in the specified directory:', load_directory, ' -> creating a new auto encoder')

            if isinstance(load_directory, str):
                base_model = AutoModelForLM.from_pretrained(load_directory, torch_dtype=torch_dtype).to(device_base_model)
                tokenizer = AutoTokenizerForLM.from_pretrained(load_directory)
            else:
                base_model = load_directory
                tokenizer = base_model.tokenizer

            layer_map = {k: v for k, v in base_model.named_parameters() if 'cls.prediction' not in k.lower() and 'lm_head' not in k.lower()}

            if layers:
                if not isinstance(layers, dict):
                    # layers information are provided
                    # check if layer description can be matched with layer_map keys
                    # layer description could also be "*.layer.10*"
                    matched_layers = []
                    for layer in layers:
                        if layer in layer_map:
                            matched_layers.append(layer)
                        else:
                            # Handle wildcard matching (e.g., "*.layer.10*")
                            layer_pattern = layer.replace('.', r'\.').replace('*', r'.*')  # Convert to regex pattern
                            layer_regex = re.compile(layer_pattern)

                            for layer_name in layer_map:
                                if layer_regex.fullmatch(layer_name):
                                    matched_layers.append(layer_name)

                    layers = list(sorted(matched_layers, key=lambda x: list(layer_map.keys()).index(x)))
            else:
                # no layer provided, i.e. all layers are used that are part of the core model, i.e. all layers that are not part of prediction layers
                layers = [layer for layer in layer_map]

            if isinstance(layers, dict):
                input_dim = sum([v.sum() for v in layers.values()])
            else:
                input_dim = sum([layer_map[layer].numel() for layer in layers])
            ae = GradiendModel(input_dim,
                               layers=layers,
                               latent_dim=latent_dim,
                               base_model=load_directory,
                               torch_dtype=torch_dtype,
                               device_encoder=device_encoder,
                               device_decoder=device_decoder,
                               **kwargs)

        # freeze all layers that do not require gradient calculations
        freeze_layers_until_target(base_model, *layers)

        model = ModelWithGradiend(base_model, ae, tokenizer, base_model_device=device_base_model, torch_dtype=torch_dtype)
        model.name_or_path = load_directory
        return model

    def ae_named_parameters(self, part='all'):
        idx = 0
        if part == 'all':
            yield from self.gradiend.named_parameters()
            return
        elif part == 'encoder':
            layer_map = {k: v for k, v in self.gradiend.encoder.named_parameters()}
            weights = layer_map['0.weight'].squeeze()
        elif 'decoder' in part:
            layer_map = {k: v for k, v in self.gradiend.decoder.named_parameters()}
            if part == 'decoder':
                weights = layer_map['0.weight'].squeeze()
            elif part == 'decoder-sum':
                weights = (layer_map['0.weight'] + layer_map['0.bias']).squeeze()
            elif part == 'decoder-bias':
                weights = layer_map['0.bias'].squeeze()
            else:
                raise ValueError('Unsupported part:', part)
        else:
            raise ValueError('Unsupported part:', part)

        for layer in self.gradiend.layers:
            orig_shape = self.layer_map[layer].shape
            num_elements = orig_shape.numel()
            yield layer, weights[idx:idx + num_elements].reshape(orig_shape)
            idx += num_elements
        if idx != weights.numel():
            raise ValueError(f'Inconsistent number of elements in the weights and expected number of elements in the layers ({idx} vs. {weights.numel()})')


    def get_top_k_neurons(self,
                          *,
                          part='decoder',
                          top_k=100,
                          scope='neuron',  # 'weight' or 'neuron'
                          scope_direction='outgoing',
                          # 'outgoing' or 'incoming' (meaningful for 'neuron' or mapping weights->neurons)
                          return_sorted=True,
                          return_format='list',  # 'list' | 'by_param' | 'idx_to_param'
                          map_to_neuron=False,  # only valid when scope == 'weight'
                          return_weight_indices=False, # when scope == 'neuron', return the weight indices that correspond to the top neurons
                          ):
        """
        Return top-k items from the specified part.

        Args:
            part: 'encoder' | 'decoder' | 'decoder_bias'
            top_k: number of top items to return
            scope: 'weight' (return top weight indices) or 'neuron' (return top neurons by aggregated weight importance)
            scope_direction: for neuron aggregation / mapping - 'outgoing' (output neurons / rows) or 'incoming' (input neurons / cols)
            return_format: 'list' (default) -> list of indices,
                           'by_param' -> dict param_name -> list,
                           'idx_to_param' -> dict index -> param_name
            map_to_neuron: if True and scope=='weight', also return mapping from weight index -> neuron index (ingoing/outgoing)

        Returns:
            Depending on return_format and map_to_neuron:
              - list of indices (default)
              - dicts as described above
              - if map_to_neuron True and return_format == 'list': returns tuple (indices_list, mapping_dict)
        """
        import numpy as np
        import torch

        assert scope in ['weight', 'neuron'], "scope must be 'weight' or 'neuron'"
        assert scope_direction in ['outgoing', 'incoming'], "scope_direction must be 'outgoing' or 'incoming'"
        assert return_format in ['list', 'by_param',
                                 'idx_to_param'], "return_format must be 'list','by_param' or 'idx_to_param'"

        # choose the correct parameter tensor
        if part == 'encoder':
            param = self.gradiend.encoder[0].weight
            param_name = 'encoder.0.weight'
            bias = getattr(self.gradiend.encoder[0], 'bias', None)
        elif part == 'decoder':
            param = self.gradiend.decoder[0].weight
            param_name = 'decoder.0.weight'
            bias = getattr(self.gradiend.decoder[0], 'bias', None)
        elif part == 'decoder_bias':
            # treat biases as a separate 1D tensor
            param = self.gradiend.decoder[0].bias
            param_name = 'decoder.0.bias'
            bias = None
        else:
            raise ValueError("part must be 'encoder', 'decoder', or 'decoder_bias'")


        if param is None:
            # nothing to do
            if return_format == 'list':
                return []
            elif return_format == 'by_param':
                return {param_name: []}
            else:
                return {}

        w = param.data.squeeze().cpu()




        if scope == 'neuron':
            # weight_to_neuron: dict(weight_idx -> neuron_identifier)
            # neuron_hit_count: dict(neuron_identifier -> count)
            weight_to_neuron, neuron_to_all_weights  = self.gradiend._get_top_k_neurons(
                self.base_model, w, top_k, scope_direction
            )

            # collect all neurons touched
            affected_neurons = list(set(weight_to_neuron.values()))

            # If the user also wants weight indices that map to each neuron:
            if return_weight_indices:
                neuron_to_weights = {nid: neuron_to_all_weights[nid] for nid in affected_neurons}
                return affected_neurons, neuron_to_weights

            # Otherwise behave like before
            return affected_neurons

        return self.gradiend.get_top_k_weights(
            part=part,
            top_k=top_k,
            scope=scope,
            scope_direction=scope_direction,
            return_sorted=return_sorted,
            return_format=return_format,
            map_to_neuron=map_to_neuron
        )

if __name__ == '__main__':
    gradiend = ModelWithGradiend.from_pretrained('results/models/bert-base-cased')