from torch import nn
from .encoder import Encoder
from .styledecoder import Synthesis
from diffusers.models.modeling_utils import ModelMixin

class Generator(ModelMixin):
    def __init__(self, size, cross_attention_dim, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)
        self.proj = nn.Linear(style_dim, cross_attention_dim)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def forward(self, img_source, img_drive, h_start=None):
        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        # img_recon = self.dec(wa, alpha, feats)
        latent = self.dec(wa, alpha, feats)
        latent = self.proj(latent)
        
        return latent
