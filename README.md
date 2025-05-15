## Installation

### **Build environment**

```
conda create -n portrait python=3.11
conda activate portrait

pip install -r requirements.txt
```

or

```bash
bash setup.sh
```

### Download models

All the pretrained models should be placed under the `./pretrained_model` directory. You can download the following models automatically using `huggingface_download.py`:

- [StableDiffusion V1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)
- [animatediff](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)

Download the pretrained checkpoint, [vox.pt](https://drive.google.com/drive/folders/1N4QcnqUQwKUZivFV-YeBuPyH4pGJHooc). It should be placed under the `./pretrained_model` directory.

Finally, these weights should be organized as follows:

```
./pretrained_model/
|-- animatediff
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
|-- stable-diffusion-v1-5
|   |-- feature_extractor
|   |   `-- preprocessor_config.json
|   |-- model_index.json
|   |-- unet
|   |   |-- config.json
|   |   `-- diffusion_pytorch_model.bin
|   `-- v1-inference.yaml
`-- vox.pt 
```

## Training

Update lines in the training config file:

```bash
data:
  root_dir: './data/vox1_png' 
```

### Stage 1

In stage 1, the Appearance Net, Denoising Net, and LIA's projection layer are trained.

Run command:

```bash
accelerate launch train_stage_1.py --config ./configs/train/stage1.yaml
```

### Stage 2

In Stage 2, only the Motion Module is trained with LDM loss, L1 loss, and VGG loss.

Run command:

```bash
accelerate launch train_stage_2.py --config ./configs/train/stage2.yaml
```

### Stage 2 (Full)

In Stage 2 (Full), the Denoising Net and LIA's projection layer are trained with LDM loss, L1 loss, and VGG loss.

Run command:

```bash
accelerate launch train_stage_2_full.py --config ./configs/train/stage2_full.yaml
```

## Evaluation

Evaluation follows the method in [this repository](https://github.com/jieun-b/portrait-eval-pipeline).

## Using DeepSpeed

```bash
pip install deepspeed
```

Modify the deepspeed_config.yaml file to fit your environment.

```bash
accelerate launch --config_file ./deepspeed_config.yaml train_<stage name>.py --config ./configs/train/<stage name>.yaml
```
