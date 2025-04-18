# 🧑‍🎨 Portrait Animation Evaluation

This repository provides a unified evaluation pipeline for comparing multiple portrait animation models under consistent settings.


## 🛠️ Build Environment

```bash
conda create -n evaluation python=3.11
conda activate evaluation

pip install -r requirements.txt
```


## 📁 Directory Structure

- `config/`: Configuration files for each model (`.yaml`)
- `checkpoint/`: Pretrained model checkpoints
- `data/`: Evaluation dataset  
  (expected structure: `data/test/{video_name}/{frame}.jpg`)
- `scripts/`
  - `inference/`: Model-specific run scripts and GT (ground truth) saving scripts
  - `metrics/`: Evaluation scripts
- `modules/`: Inference logic per model
- `eval/`: Output directory for evaluation results  
  (saved in `eval/{reconstruction, animate}/{model_name}`)


## 🚀 Run Inference

```bash
python -m scripts.inference.<model> --mode reconstruction --config config/<model>.yaml --checkpoint checkpoint/<model>.pth
python -m scripts.inference.<model> --mode animation --config config/<model>.yaml --checkpoint checkpoint/<model>.pth
```


## 📊 Run Evaluation

### Reconstruction

```bash
python -m scripts.metrics.reconstruction_eval --gen_dirs fomm fvv lia
```

This script computes L1, LPIPS, SSIM, PSNR, AKD, and AED metrics between generated results and ground truth.

### Animation

```bash
python -m scripts.metrics.animation_eval --gen_dirs fomm fvv lia
```

This script computes FID and CSIM between generated results and source frame.


## 🔧 Dataset Configuration

When running in `animation` mode, make sure the following value is set in your YAML config file:

```bash
dataset_params:
  is_full: False
```


## 💾 Result Format

```bash
eval/
├── animation/
│   ├── gt/
│   │   ├── source/
│   │   └── driving/
│   ├── fomm/
│   │   ├── driving-source/000.png, 001.png, ...
│   │   └── compare/driving-source.gif
│   ├── fvv/
│   ├── lia/
│   └── metrics.json
└── reconstruction/
    ├── gt/
    ├── fomm/
    │   ├── <name1>/000.png, 001.png, ...
    │   └── compare/<name1>.gif
    ├── fvv/
    ├── lia/
    └── metrics.json
```


## 🔗 References

- [First Order Motion Model for Image Animation](https://github.com/AliaksandrSiarohin/first-order-model), presented at NIPS 2019.
- [One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), presented at CVPR 2021.
- [Latent Image Animator: Learning to Animate Images via Latent Space Navigation](https://github.com/wyhsirius/LIA), presented at ICLR 2022.