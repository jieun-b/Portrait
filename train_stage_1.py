import argparse
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

from datetime import datetime
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
from diffusers.models import UNet2DConditionModel 
from transformers import CLIPVisionModelWithProjection

from src.dataset.dataset import FramesDataset, ValidDataset, collate_fn, DatasetRepeater
from src.models.net import Net
from src.models.LIA.generator import Generator
from src.models.mutual_self_attention import ReferenceAttentionControl 
from src.pipelines.pipeline_img2img import Image2ImagePipeline 
from src.utils.util import seed_everything, save_checkpoint, delete_additional_ckpt, import_filename, compute_snr

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
    valid_dataset
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    appearance_unet = ori_net.appearance_unet
    denoising_unet = ori_net.denoising_unet
    lia = ori_net.lia
    
    generator = torch.manual_seed(42)
    
    pipe = Image2ImagePipeline(
        vae=vae,
        image_encoder=image_encoder,
        appearance_unet=appearance_unet,
        denoising_unet=denoising_unet,
        lia=lia,
        scheduler=scheduler,
    ).to(accelerator.device)

    pil_images = []

    dataset_len = len(valid_dataset)
    sample_idx = [np.random.randint(0, dataset_len-1) for _ in range(3)]
    
    for idx in sample_idx:
        sample = valid_dataset[idx]
        ref_image_pil = Image.fromarray(sample['src_img']).convert("RGB")
        gt_image_pil = Image.fromarray(sample['tar_gt']).convert("RGB")
        
        image = pipe(
            ref_image_pil,
            gt_image_pil,
            width,
            height,
            num_inference_steps=25,
            guidance_scale=3.5,
            generator=generator,
        ).images
        image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, h, w)
        res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
        # Save ref_image, gt_image and the generated_image
        w, h = res_image_pil.size
        canvas = Image.new("RGB", (w * 3, h), "white")
        ref_image_pil = ref_image_pil.resize((w, h))
        gt_image_pil = gt_image_pil.resize((w, h))

        canvas.paste(ref_image_pil, (0, 0))
        canvas.paste(res_image_pil, (w, 0))
        canvas.paste(gt_image_pil, (w * 2, 0))

        pil_images.append({"name": f"sample_{idx}", "img": canvas})

    del pipe
    torch.cuda.empty_cache()

    return pil_images


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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
    appearance_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet")
    denoising_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path)
    lia = Generator(256, denoising_unet.config.cross_attention_dim)
    lia.load_state_dict(torch.load(cfg.lia_model_path, map_location="cpu")["gen"], strict=False)
    
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    cond_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    
    for model in [vae, image_encoder, appearance_unet, denoising_unet, lia]:
        model.to(accelerator.device)
        if cfg.solver.fp16_mode:
            model.half()
        elif model in [vae, image_encoder]:
            model.to(dtype=weight_dtype)

        if model in [vae, image_encoder]:
            model.requires_grad_(False)
        else:
            model.requires_grad_(True)

    # Enable gradients for specific layers
    for name, param in lia.named_parameters():
        param.requires_grad = "proj" in name
            
    for name, param in appearance_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad = False
            
    do_classifier_free_guidance = False

    reference_control_writer = ReferenceAttentionControl(
        appearance_unet,
        do_classifier_free_guidance=do_classifier_free_guidance,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=do_classifier_free_guidance,
        mode="read",
        fusion_blocks="full",
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
        
    train_dataset = FramesDataset(**cfg.data, is_image=True)
    valid_dataset = ValidDataset(**cfg.data, is_image=True)
    
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
            project_name="project_name", 
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
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values = batch["pixel_values_tar"]
                if cfg.solver.fp16_mode:
                    pixel_values = pixel_values.half()
                else:
                    pixel_values = pixel_values.to(dtype=weight_dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1),
                        device=noise.device,
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
                    lia_tar_image = cond_image_processor.preprocess(tar_img.to(dtype=lia.dtype, device=lia.device), height=256, width=256)
                    
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

                model_pred = net(
                    noisy_latents,
                    timesteps,
                    src_image_latents,
                    clip_src_embeds,
                    lia_src_image,
                    lia_tar_image,
                    uncond_fwd,
                )
    
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

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
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
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
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
                            valid_dataset=valid_dataset,
                        )
                        
                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            img = sample_dict["img"]
                            out_file = os.path.join(sample_dir, f'{global_step:06d}-{sample_name}.png')
                            img.save(out_file)
                                
                        reference_control_writer = ReferenceAttentionControl(
                            appearance_unet,
                            do_classifier_free_guidance=False,
                            mode="write",
                            fusion_blocks="full",
                        )
                        reference_control_reader = ReferenceAttentionControl(
                            denoising_unet,
                            do_classifier_free_guidance=False,
                            mode="read",
                            fusion_blocks="full",
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
            save_checkpoint(unwrap_net.appearance_unet, save_dir, "appearance_unet", global_step, total_limit=3)
            save_checkpoint(unwrap_net.denoising_unet, save_dir, "denoising_unet", global_step, total_limit=3)
            save_checkpoint(unwrap_net.lia, save_dir, "lia", global_step, total_limit=3)
    
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage1.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)