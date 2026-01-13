import json
import os
from typing import List
import torch
import torch.nn as nn
from gradiend.model import GradiendModel, LargeLinear, ModelWithGradiend



class StackedGradiend(GradiendModel):
    def __init__(self, input_dim, layers, models, latent_dim=1, *args, **kwargs):
        super().__init__(input_dim=input_dim, latent_dim=1, layers=layers, *args, **kwargs)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for model in models:
            self.encoders.append(model.gradiend.encoder)
            self.decoders.append(model.gradiend.decoder)

        # in any case freeze encoder weights I do not want to retrain the encoder part, only the decoder if needed.
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = False

        # TODO find a better way to
        self.source = models[0].gradiend.kwargs["training"]["source"]
        # TODO find a better way to do this..
        self.base_model = models[0].base_model

        self.models = models
        self.device = self.base_model.device

    def encode(self, x):
        encoded = []
        for encoder in self.encoders:
            out = encoder(x).item()
            encoded.append(out)
        return encoded

    def decode(self, x):
        # from the deocoder I want ONE iutput...
        # 'sum' or average'
        decoded = []
        for decoder in self.decoders:
            out = decoder(out)
            decoded.append(out)
        return decoded

    def modify_model_decode(self, x):
        decoded = []
        for decoder in self.decoders:
            out = decoder(torch.tensor([x], dtype=torch.float, device=self.device))
            decoded.append(out)
        return torch.mean(torch.stack(decoded), dim=0)

    def modify_model_decode_v1(self, x):
        decoded = []
        for decoder in self.decoders:
            out = decoder(torch.tensor([x], dtype=torch.float, device=self.device))
            decoded.append(out)
        return decoded

    def forward(self, x, return_encoded=False):
        pass

    def extract_gradients(self, bert, return_dict=False):
        return super().extract_gradients(self.base_model, return_dict)



def _get_model_layers(models_with_grad: List[str]):
    if not models_with_grad:
        raise ValueError("No models were provided")

    models = _load_grad_models(models_with_grad)

    ordered_lists = []
    for model in models:
        # keep original order from named_parameters()
        names = [k for k, _ in model.base_model.named_parameters()
                 if "cls.prediction" not in k.lower()]
        ordered_lists.append(names)


    ref = set(ordered_lists[0])
    if all(set(lst) == ref for lst in ordered_lists[1:]):
        return ordered_lists[0]   
    else:
        raise ValueError("Models have different parameter name sets")


def _get_input_dim(models_with_grad: List['str']): 
    if not models_with_grad: 
        raise ValueError('No models were provided')
    
    models_with_grad = _load_grad_models(models_with_grad)
    
    input_dims = []

    for model in models_with_grad: 
        input_dim = getattr(model.gradiend, "input_dim", None)
        input_dims.append(input_dim)
    
    #input_dims = [getattr(model.gradiend, "input_dim", None) for model in models_with_grad]


    if all(dim == input_dims[0] for dim in input_dims[1:]):
        return input_dims[0]
    else: 
        raise ValueError('Models do not have the same input dimensions. Cannot combine them.')

def _load_grad_models(model_paths: List['str']) -> List['ModelWithGradiend']:
    grad_models = []
    for model_path in model_paths: 
        grad_model = ModelWithGradiend.from_pretrained(model_path)
        grad_models.append(grad_model)
    return grad_models

 
class MultiEncoder(nn.Module):
    def __init__(self, encoders, mode: str = "concat", freeze = False):
        super().__init__()
        if freeze: 
            for encoder in encoders:
                for param in encoder.parameters():
                    param.requires_grad = False

        self.encoders = nn.ModuleList(encoders)
        assert mode in ("concat", "stack")
        self.mode = mode

    def forward(self, x):
        outs = [enc(x) for enc in self.encoders]
        return torch.cat(outs, dim=-1) if self.mode == "concat" else torch.stack(outs, dim=-1)
    


    @property
    def encoder_norm(self):
        total_sq = 0.0
        for enc in self.encoders:  # MultiEncoder.encoders is your ModuleList
            total_sq += torch.norm(enc[0].weight, p=2).item() ** 2
        return total_sq ** 0.5

    

class CombinedEncoderDecoder(GradiendModel):
    def __init__(
        self,
        grad_model_paths: List['str'],
        num_encoders,
        latent_dim,
        freeze_encoder=False,
        activation="tanh",
        dec_init="trained",
        merge=False,
        shared=False,
        decoder_factor=1.0,
        **kwargs,
    ):
        super().__init__(
            input_dim=_get_input_dim(grad_model_paths) if grad_model_paths else None,
            latent_dim=latent_dim,
            layers=_get_model_layers(grad_model_paths) if grad_model_paths else None,
            activation=activation,
            **kwargs,
        )

        self.grad_model_paths = grad_model_paths
        self.num_encoders = len(grad_model_paths) if grad_model_paths else None
        self.dec_init = dec_init
        self.input_dim = _get_input_dim(grad_model_paths) if grad_model_paths else None
        self.layers = _get_model_layers(grad_model_paths) if grad_model_paths else None
        self.freeze_encoder = freeze_encoder
        self.decoder_factor = decoder_factor
        self.latent_dim = latent_dim
        self.activation = activation
        self.shared = shared
        self.kwargs = kwargs 
        self.merge = merge
        self.grad_models = _load_grad_models(grad_model_paths)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.merge and self.grad_models: # keeps the encoders and decoders separate
            self.encoder = MultiEncoder([m.gradiend.encoder for m in self.grad_models], mode="concat", freeze=freeze_encoder)
            self.decoder = self._set_up_decoder(self.grad_models, dec_init=self.dec_init)

               
        else:
            encoders = nn.ModuleList()
            self.encoder = MultiEncoder(encoders, mode="concat")

            if self.grad_models:
                self.og_decoders = nn.ModuleList()
                self.og_encoders = nn.ModuleList()

                for model in self.grad_models:
                    self.og_decoders.append(model.gradiend.decoder)
                    self.og_encoders.append(model.gradiend.encoder)

                with torch.no_grad():
                    encoder_states = []
                    for encoder in self.og_encoders:
                        encoder_states.append(encoder.state_dict())

                    for i, state in enumerate(encoder_states):
                        print(f"Encoder {i}:")
                        for name, tensor in state.items():
                            print(f"  {name}: {tensor.shape}")

                    new_state = {}
                    state_dict = self.grad_models[0].gradiend.encoder.state_dict()
                    state_dict_1 = self.grad_models[1].gradiend.encoder.state_dict()
                    state_dict_2 = self.grad_models[2].gradiend.encoder.state_dict()

                    print(state_dict.keys(), state_dict_1.keys(), state_dict_2.keys())

                    for key in state_dict.keys():
                        if "linear" in key and "weight" in key:
                            new_state[key] = torch.cat(
                                [encoder_state[key] for encoder_state in encoder_states],
                                dim=0,
                            )
                        elif "linear" in key and "bias" in key:
                            bias_avg = sum(encoder_state[key] for encoder_state in encoder_states) / len(
                                encoder_states
                            )
                            new_bias = torch.full((3,), bias_avg.item())

                            new_state[key] = new_bias.clone()

                self.encoder.load_state_dict(new_state)
         


    @property
    def encoder_norm(self):
        if not self.merge:
           return self.encoder.encoder_norm
        else:
            return torch.norm(self.encoder[0].weight, p=2).item()

    @property
    def decoder_norm(self):
        return torch.norm(self.decoder[0].weight, p=2).item()

    def _set_up_decoder(self, grad_models: List['ModelWithGradiend'], dec_init='trained' ):
        decoder = nn.Sequential(LargeLinear(self.num_encoders, self.input_dim, device=self.device), nn.Tanh())

        if dec_init == "trained":
            with torch.no_grad():
                decoder_states = []
                for model in grad_models:
                    decoder_states.append(model.gradiend.decoder.state_dict())


                new_state = {}
                 
                state_dict = grad_models[0].gradiend.decoder.state_dict()
            
                for key in state_dict.keys():
                    if "linear" in key and "weight" in key:
                        new_state[key] = torch.cat(
                            [decoder_state[key] for decoder_state in decoder_states],
                            dim=1,
                        )
                    elif "linear" in key and "bias" in key:
                        new_state[key] = sum(decoder_state[key] for decoder_state in decoder_states) / len(
                            decoder_states
                        )


            decoder.load_state_dict(new_state)

        elif dec_init == "scratch":
            # for i in range(num_encoders):
            #     x = self.encoder[i][0].weight.max().item() * self.decoder_factor
            #     self.encoder_scales.append(x)

            with torch.no_grad():
                nn.init.xavier_uniform_(decoder[0].weight)
                # for i, scale in enumerate(self.encoder_scales):
                #     nn.init.uniform_(self.decoder[0].weight[:, i:i+1], -scale, scale)
        else:
            NotImplementedError

        return decoder


    @classmethod
    def from_pretrained(cls, load_path, device=None):
        model_path  = os.path.join(load_path, "pytorch_model.bin")
        config_path = os.path.join(load_path, "config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

   
        model = cls(**config)

       
        state = torch.load(model_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=True) 

        if missing or unexpected:
            print("[load] missing:", missing)
            print("[load] unexpected:", unexpected)

        model.name_or_path = load_path

        if 'layers_path' in config:
            layers_path = os.path.join(load_path, config['layers_path'])
            try:
                model.layers = torch.load(layers_path, map_location=device)
            except FileNotFoundError:
                print(f"Warning: {layers_path} not found.")

        return model

    
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
            'grad_model_paths': self.grad_model_paths,
            'latent_dim': self.latent_dim,
            'activation': self.activation,
            'activation_decoder': self.activation_decoder,
            'bias_decoder': self.bias_decoder,
            'merge': self.merge,
            'freeze_encoder': self.freeze_encoder,
            'activation': self.activation,
            'dec_init': self.dec_init,
            'shared': self.shared,
            'num_encoders': self.num_encoders,
            **self._serialize_kwargs(),
        }

        if isinstance(self.layers, dict):
            config['layers_path'] = 'layers.pth'

        with open(config_path, 'w') as f:
            json.dump(config, f)

    def _serialize_kwargs(self):
        kwargs = self.kwargs.copy()
        training_kwargs = kwargs['training'].copy()

        # if training_kwargs['layers'] is not None and isinstance(training_kwargs['layers'], dict):
        #     training_kwargs['layers'] = list(training_kwargs['layers'].keys())
        #     training_kwargs['layers_path'] = 'layers.pth'
        kwargs['training'] = training_kwargs

        return kwargs

    def encode(self, x):
        return self.encoder(x)

    def modified_decode(self, x, method="sum"):
        out = self.decoder(torch.tensor(x, dtype=torch.float, device=self.device))

        if method == "sum":
            return out.squeeze(0)
        else:
            return torch.mean(out, dim=1)

    def forward(self, x, return_encoded=False):
        return super().forward(x, return_encoded)

