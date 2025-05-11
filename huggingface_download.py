from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse", 
    local_dir="./pretrained_model/sd-vae-ft-mse"
)

snapshot_download(
    repo_id="guoyww/animatediff-motion-adapter-v1-5-2", 
    local_dir="./pretrained_model/animatediff", 
)

snapshot_download(
    repo_id="lambdalabs/sd-image-variations-diffusers", 
    local_dir="./pretrained_model", 
    allow_patterns=["image_encoder/*"]
)

snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir="./pretrained_model/stable-diffusion-v1-5",
    allow_patterns=[
        "feature_extractor/preprocessor_config.json", 
        "model_index.json", 
        "unet/config.json", 
        "unet/diffusion_pytorch_model.safetensors",
        "v1-inference.yaml"
    ],
)