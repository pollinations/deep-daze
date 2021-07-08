##Working further with Siren, to experiment deeper with the network and see what really clicks.
##Most of this taken from the siren-pytorch library, because I can't actually code lmao

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from mfn import FourierNet, GaborNet

from .utils import exists, enable

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

#Custom activation. Will it work? ¯\_(ツ)_/¯


class MFN(nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
       

        self.image_width = image_width
        self.image_height = image_height

        self.model = GaborNet(
            in_size=2,
            hidden_size=256,
            out_size=3,
            n_layers=3,
            input_scale=256,
            weight_scale=1,
        )

        tensors = [torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height)]
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')

        self.register_buffer('grid', mgrid)
        # self.coords = torch.stack(
        #             torch.meshgrid(
        #                 [
        #                     torch.linspace(-1.0, 1.0, image_width),
        #                     torch.linspace(-1.0, 1.0, image_height),
        #                 ]
        #             ),
        #             dim=-1,
        #         ).view(-1, 2)
    def forward(self, img = None, *, latent = None):

        coords = self.grid.clone().detach().requires_grad_()
        out = self.model(coords)
        print("pre shape",out.shape)
        out = (out*0.5 + 0.5).clamp(0,1)
        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)
        print("post shape",out.shape)
        if exists(img):
            return F.mse_loss(img, out)

        return out