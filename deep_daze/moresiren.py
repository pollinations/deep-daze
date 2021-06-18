##Working further with Siren, to experiment deeper with the network and see what really clicks.
##Most of this taken from the siren-pytorch library, because I can't actually code lmao

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from siren_pytorch import Sine, Siren

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

#Custom activation. Will it work? ¯\_(ツ)_/¯

class CustomActivation(nn.Module):
	def __init__(self, torch_activation=torch.sin, w0 = 1.):
		super().__init__()
		self.w0 = w0
		self.activation = torch_activation
	def forward(self, x):
		return self.activation(self.w0 * x)

#because I don't wanna do 2 repos, here's a more "open" SirenNet class, and by that I mean just changing activations on the layers themselves lol
class CustomSirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, layer_activation = None, final_activation = None):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                activation = layer_activation
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)