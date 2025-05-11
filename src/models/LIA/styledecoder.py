import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Direction(nn.Module):
    def __init__(self, motion_dim):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(512, motion_dim))
        
    def forward(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight.float())  # get eignvector, orthogonal [n1, n2, n3, n4]
        
        if input is None:
            return Q
        else:
            Q = Q.to(input.dtype) 
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out


class Synthesis(nn.Module):
    def __init__(self, size, style_dim, motion_dim, blur_kernel=[1, 3, 3, 1], channel_multiplier=1):
        super(Synthesis, self).__init__()

        self.size = size
        self.style_dim = style_dim
        self.motion_dim = motion_dim

        self.direction = Direction(motion_dim)

    def forward(self, wa, alpha, feats):

        # wa: bs x style_dim
        # alpha: bs x style_dim

        bs = wa.size(0)

        if alpha is not None:
            # generating moving directions
            if len(alpha) > 1:
                directions_target = self.direction(alpha[0])  # target
                directions_source = self.direction(alpha[1])  # source
                directions_start = self.direction(alpha[2])  # start
                latent = wa + (directions_target - directions_start) + directions_source
            else:
                directions = self.direction(alpha[0])
                latent = wa + directions  # wa + directions
        else:
            latent = wa

        return latent
