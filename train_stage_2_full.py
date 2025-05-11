import argparse
import copy
import logging
import math
import os
import random
import warnings
import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from datetime import datetime
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import UNet2DConditionModel, MotionAdapter
from transformers import CLIPVisionModelWithProjection

from src.dataset.dataset import FramesDataset, ValidDataset, collate_fn, DatasetRepeater
from src.models.net import Net
from src.models.LIA.generator import Generator
from src.models.mutual_self_attention import ReferenceAttentionControl 
from src.models.unet_motion_model import UNetMotionModel
from src.pipelines.pipeline_vid2vid import Video2VideoPipeline
from src.utils.util import (
    seed_everything, save_checkpoint, delete_additional_ckpt, import_filename, save_videos_grid, 
    compute_snr, predict_xstart, decode_latents
)

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def log_validation(
    vae,
    image_encoder,
    net,
    scheduler,
    accelerator,
    width,
    height,
    clip_length=24,
    generator=None,
    valid_dataset=None
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    appearance_unet = ori_net.appearance_unet
    denoising_unet = ori_net.denoising_unet
    lia = ori_net.lia

    if generator is None:
        generator = torch.manual_seed(42)
    tmp_denoising_unet = copy.deepcopy(denoising_unet)
    tmp_denoising_unet = tmp_denoising_unet.to(dtype=denoising_unet.dtype)

    pipe = Video2VideoPipeline(
        vae=vae,
        image_encoder=image_encoder,
        appearance_unet=appearance_unet,
        denoising_unet=tmp_denoising_unet,
        lia=lia,
        scheduler=scheduler,
    ).to(accelerator.device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((height, width)),
        torchvision.transforms.ToTensor()
    ])

    results = []
    sample_idx = [np.random.randint(0, len(valid_dataset)-1) for _ in range(3)]
    for idx in sample_idx:
        sample = valid_dataset[idx]
        ref_image_pil = Image.fromarray(sample["src_img"]).convert("RGB")
        gt_images = [Image.fromarray(img).convert("RGB") for img in sample["tar_gt"][:clip_length]]
        
        ref_tensor = torch.stack([transform(ref_image_pil)] * len(gt_images)).transpose(0, 1)  # (c, f, h, w)
        gt_tensor = torch.stack([transform(img) for img in gt_images]).transpose(0, 1)         # (c, f, h, w)

        pipeline_output = pipe(
            ref_image_pil,
            gt_images,
            width,
            height,
            clip_length,
            25,
            3.5,
            generator=generator,
        )
        video = pipeline_output.videos

        # Concat it with pose tensor
        ref_tensor = ref_tensor.unsqueeze(0)
        gt_tensor = gt_tensor.unsqueeze(0)
        video = torch.cat([ref_tensor, video, gt_tensor], dim=0)

        results.append({"name": f"sample_{idx}", "vid": video})

    del tmp_denoising_unet
    del pipe
    torch.cuda.empty_cache()

    return results


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    amp_mode = cfg.solver.mixed_precision if not cfg.solver.fp16_mode else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=amp_mode,
        kwargs_handlers=[kwargs],
        # log_with='wandb'
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = os.path.join(cfg.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    sample_dir = os.path.join(save_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    weight_dtype = torch.float16 if cfg.solver.fp16_mode or cfg.weight_dtype == "fp16" else torch.float32

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path)
    motion_adapter = MotionAdapter.from_config(MotionAdapter.load_config(cfg.motion_adapter_path))
    appearance_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet")
    denoising_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet")
    denoising_unet = UNetMotionModel.from_unet2d(denoising_unet, motion_adapter)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path)
    lia = Generator(256, denoising_unet.config.cross_attention_dim)
    vgg19 = torchvision.models.vgg19(pretrained=True).features[:16].eval()

    denoising_unet.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"))
    appearance_unet.load_state_dict(torch.load(cfg.appearance_unet_path, map_location="cpu"))
    lia.load_state_dict(torch.load(cfg.lia_model_path, map_location="cpu"))

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    cond_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False)
    
    for model in [vgg19, vae, image_encoder, appearance_unet, denoising_unet, lia]:
        model.to(accelerator.device)
        if cfg.solver.fp16_mode:
            model.half()
        elif model in [vgg19, vae, image_encoder, appearance_unet]:
            model.to(dtype=weight_dtype)
        model.requires_grad_(False)
                
    denoising_unet.requires_grad_(True)
    for name, param in lia.named_parameters():
        param.requires_grad = "proj" in name

    reference_control_writer = ReferenceAttentionControl(
        appearance_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
        video_length=cfg.data.sample_n_frames,
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
        video_length=cfg.data.sample_n_frames,
    )

    net = Net(
        appearance_unet,
        denoising_unet,
        lia,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            appearance_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        appearance_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()
        vae.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = FramesDataset(**cfg.data, is_image=False)
    valid_dataset = ValidDataset(**cfg.data, is_image=False)
    
    if cfg.solver.num_repeats != 1:
        train_dataset = DatasetRepeater(train_dataset, cfg.solver.num_repeats)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.train_bs, 
        shuffle=True, 
        num_workers=16,
        drop_last=True,
        collate_fn=collate_fn,
    )


    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d")
        accelerator.init_trackers(
            project_name='project_name', 
            config={
                "learning_rate": cfg.solver.learning_rate,
                "max_train_steps": cfg.solver.max_train_steps,
                "batch_size": cfg.train_bs,
            }
        )

    # Train!
    total_batch_size = (
        cfg.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss, train_ldm_loss, train_vgg_loss, train_l1_loss = 0.0, 0.0, 0.0, 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values_vid = batch["pixel_values_tar"]
                if cfg.solver.fp16_mode:
                    pixel_values_vid = pixel_values_vid.half()
                else:
                    pixel_values_vid = pixel_values_vid.to(dtype=weight_dtype)

                with torch.no_grad():
                    video_length = pixel_values_vid.shape[1]
                    pixel_values_vid = rearrange(
                        pixel_values_vid, "b f c h w -> (b f) c h w"
                    )
                    latents = vae.encode(pixel_values_vid).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                uncond_fwd = random.random() < cfg.uncond_ratio
                
                src_image_list = []
                tar_image_list = []
                processed_src_list = []

                for batch_idx, (src_img, tar_img, processed_src) in enumerate(
                    zip(
                        batch["pixel_values_src"],
                        batch["pixel_values_tar"],
                        batch["src_image"],
                    )
                ):
                    if uncond_fwd:
                        processed_src_list.append(torch.zeros_like(processed_src))
                    else:
                        processed_src_list.append(processed_src)
                    src_image_list.append(src_img)
                    tar_image_list.append(tar_img)

                with torch.no_grad():
                    src_img = torch.stack(src_image_list, dim=0)
                    tar_img = torch.stack(tar_image_list, dim=0)
                    
                    src_image_latents = vae.encode(src_img.to(dtype=vae.dtype, device=vae.device)).latent_dist.sample()  # (bs, d, 64, 64)
                    src_image_latents = src_image_latents * vae.config.scaling_factor

                    src_image = torch.stack(processed_src_list, dim=0).to(dtype=image_encoder.dtype, device=image_encoder.device)
                    clip_src_embeds = image_encoder(src_image).image_embeds.unsqueeze(1)
                    
                    lia_src_image = cond_image_processor.preprocess(src_img.to(dtype=lia.dtype, device=lia.device), height=256, width=256)
                    lia_tar_image_list = []
                    for i in range(tar_img.shape[1]):
                        lia_tar_image = cond_image_processor.preprocess(
                            tar_img[:, i].to(dtype=lia.dtype, device=lia.device), height=256, width=256
                        ).unsqueeze(2) 
                        lia_tar_image_list.append(lia_tar_image)
                    lia_tar_images = torch.cat(lia_tar_image_list, dim=2)

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )
                
                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )
                
                # ---- Forward!!! -----
                model_pred = net(
                    noisy_latents,
                    timesteps,
                    src_image_latents,
                    clip_src_embeds,
                    lia_src_image,
                    lia_tar_images,
                    uncond_fwd=uncond_fwd,
                )
                
                mask = timesteps <= cfg.time_threshold  # (bsz,)
                pred_x_0, gt_x_0 = None, None 
                if mask.any():  
                    pred_z_0 = predict_xstart(train_noise_scheduler, timesteps[mask], model_pred[mask], noisy_latents[mask])
                    pred_x_0 = decode_latents(vae, pred_z_0)
                    
                    pred_x_0_ = rearrange(pred_x_0, "b c f h w -> (b f) c h w")
                    pred_x_0_ = F.interpolate(pred_x_0_, size=(224, 224), mode="bilinear")
                    
                    pred_feats_vgg = vgg19(pred_x_0_)

                    with torch.no_grad():
                        gt_z_0 = predict_xstart(train_noise_scheduler, timesteps[mask], target[mask], noisy_latents[mask])
                        gt_x_0 = decode_latents(vae, gt_z_0)
                        
                        gt_x_0_ = rearrange(gt_x_0, "b c f h w -> (b f) c h w")
                        gt_x_0_ = F.interpolate(gt_x_0_, size=(224,224), mode="bilinear")
        
                        gt_feats_vgg = vgg19(gt_x_0_)
                        
                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                if pred_x_0 is not None and gt_x_0 is not None:
                    l1_loss = F.l1_loss(pred_x_0, gt_x_0, reduction="mean")
                    vgg_loss = F.l1_loss(pred_feats_vgg, gt_feats_vgg, reduction="mean")
                else:
                    l1_loss = torch.tensor(1e-6, requires_grad=True, device=model_pred.device)
                    vgg_loss = torch.tensor(1e-6, requires_grad=True, device=model_pred.device)
                    
                total_loss = loss + l1_loss * cfg.l1_scale + vgg_loss * cfg.vgg_scale 

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_ldm_loss = accelerator.gather(loss.repeat(cfg.train_bs)).mean()
                avg_l1_loss = accelerator.gather(l1_loss.repeat(cfg.train_bs)).mean()
                avg_vgg_loss = accelerator.gather(vgg_loss.repeat(cfg.train_bs)).mean()
                avg_total_loss = accelerator.gather(total_loss.repeat(cfg.train_bs)).mean()
                
                train_loss += avg_total_loss.item() / cfg.solver.gradient_accumulation_steps
                train_ldm_loss += avg_ldm_loss.item() / cfg.solver.gradient_accumulation_steps
                train_l1_loss += avg_l1_loss.item() / cfg.solver.gradient_accumulation_steps
                train_vgg_loss += avg_vgg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "ldm_loss": train_ldm_loss, "vgg_loss": train_vgg_loss, "l1_loss": train_l1_loss}, step=global_step)
                train_loss, train_ldm_loss, train_vgg_loss, train_l1_loss = 0.0, 0.0, 0.0, 0.0
                if global_step % cfg.checkpointing_steps == 0 and (not isinstance(net, deepspeed.DeepSpeedEngine) or accelerator.is_main_process):
                    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    delete_additional_ckpt(save_dir, 2)
                    accelerator.save_state(save_path)
                        
                if (global_step % cfg.val.validation_steps == 0) or (global_step in cfg.val.validation_steps_tuple):
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            vae=vae,
                            image_encoder=image_encoder,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.sample_size[0],
                            height=cfg.data.sample_size[1],
                            clip_length=cfg.data.sample_n_frames,
                            generator=generator,
                            valid_dataset=valid_dataset
                        )

                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            vid = sample_dict["vid"]
                            out_file = os.path.join(sample_dir, f'{global_step:06d}-{sample_name}.gif')
                            save_videos_grid(vid, out_file, n_rows=4)

                        reference_control_writer = ReferenceAttentionControl(
                            appearance_unet,
                            do_classifier_free_guidance=False,
                            mode="write",
                            fusion_blocks="full",
                            video_length=cfg.data.sample_n_frames,
                        )
                        reference_control_reader = ReferenceAttentionControl(
                            denoising_unet,
                            do_classifier_free_guidance=False,
                            mode="read",
                            fusion_blocks="full",
                            video_length=cfg.data.sample_n_frames,
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
            
        # save model after each epoch
        if (epoch + 1) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(unwrap_net.denoising_unet, save_dir, "denoising_unet", global_step, total_limit=3)
            save_checkpoint(unwrap_net.lia, save_dir, "lia", global_step, total_limit=3)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2_full.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)