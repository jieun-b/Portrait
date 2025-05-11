import torch
import torch.nn as nn
from typing import Union

from diffusers.models import UNet2DConditionModel

from src.models.unet_motion_model import UNetMotionModel

class Net(nn.Module):
    def __init__(
        self,
        appearance_unet: UNet2DConditionModel,
        denoising_unet: Union[UNet2DConditionModel, UNetMotionModel],
        lia,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.appearance_unet = appearance_unet
        self.denoising_unet = denoising_unet
        self.lia = lia
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        src_image_latents,
        clip_src_embeds,
        lia_src_image,
        lia_tar_images,
        uncond_fwd: bool = False,
    ):
        if lia_tar_images.ndim != 5:
            lia_latent_embeds = self.lia(
                lia_src_image, 
                lia_tar_images
            )
            lia_latent_embeds = lia_latent_embeds.unsqueeze(1)
        else:
            latent_embeds_list = []
            for i in range(lia_tar_images.shape[2]):
                lia_latent_embeds = self.lia(lia_src_image, lia_tar_images[:, :, i])
                latent_embeds_list.append(lia_latent_embeds)
            lia_latent_embeds = torch.cat(latent_embeds_list, dim=0).unsqueeze(1)

        if not uncond_fwd:
            src_timesteps = torch.zeros_like(timesteps)
            self.appearance_unet(
                src_image_latents,
                src_timesteps,
                encoder_hidden_states=clip_src_embeds,
                return_dict=False,
            )
            # w.bank -> r.bank, w.bank clear
            self.reference_control_reader.update(self.reference_control_writer) 
            
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=lia_latent_embeds,
        ).sample
        
        return model_pred