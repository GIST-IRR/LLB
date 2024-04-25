# [ICLR 2024]Label-focused Latent-object Biasing (LLB) Implementation
 
This repository contains implementation of the paper [Label-Focused Inductive Bias over Latent Object Features in Visual Classification](https://openreview.net/forum?id=cH3oufN8Pl&referrer=%5Bthe%20profile%20of%20Ilmin%20Kang%5D(%2Fprofile%3Fid%3D~Ilmin_Kang1)) published on  ICLR 2024

<p align="center">
  <img src='https://github.com/IlminKang/LLB/blob/master/images/overview_all.jpg' alt="overview" width="520" >
</p>

We use pytorch Multi-processing Distributed Data Parallel Training for training Label-focused Latent-object Biasing (LLB) method

LLB takes visual features from Vision Transformer (ViT) and proceeds the following steps,
 - First, learns intermediate latent object features in an unsupervised manner,
 - decouples their visual dependencies by assigning new independent embedding parameters,
 - it captures structured features optimized for the original classification task,
 - it integrates the structured features with the original visual features for final,
prediction

## Dataset
- ImageNet
- Places365 
- iNaturalist2018


## Backbone

- ViT (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale) [[paper]](https://arxiv.org/abs/2010.11929) [[git]](https://github.com/huggingface/pytorch-image-models)
- MAE (Masked Autoencoders Are Scalable Vision Learners) [[paper]](https://arxiv.org/abs/2111.06377) [[git]](https://github.com/facebookresearch/mae)
- SWAG (Revisiting Weakly Supervised Pre-Training of Visual Perception Models) [[paper]](https://arxiv.org/abs/2201.08371) [[git]](https://github.com/facebookresearch/SWAG)

## Requirements
- Pre-trained weights of different backbones (check the link at the Backbone section)
- Save path to the pre-trained weights in `/models/__inti__.py`
- Also, save path to the data in `/utils/general.py`

## Environment
- torch==1.12.0
- torchvision==0.13.0
- timm==0.9.2
- numpy==1.21.5
- A100 GPUs * 8

## Train
To pre-train ViT-Base with multi-processing distributed training, run the following codes.
```bash
torchrun --nproc_per_node=4 train.py \
--amp \
--seed 123 \
--save \
--method timm_augreg_in21k_ft_in1k \
--encoder ViT \
--vit_size Base \
--transfer \
--freeze \
--alpha 0.8 \
--num_nvit_layers 6 \
--target_layer 11 \
--object_size 2048 \
--dataset ImageNet \
--batch_size 128 \
--epochs 70 \
--lr_scheduler CosineAnnealingLR \
--opt Adam 
```
- We run the model in multi-GPUs using multi-processing distributed using pytorch native Distributed Data Parallel (DDP)
- Set `--dataset` to `imageet` if train LLB with ImageNet1K dataset
  - For others, use `places365` or `inaturalist2018`
- To use ImageNet21K pre-trained ViT for backbone, use 'timm_augreg_in21k_ft_in1k' for `--method`
  - For SWAG, use `swag_ig_ft_plc365`, `swag_ig_ft_in1k`, `swag_ig_ft_inat18` for `--method`
  - for MAE, use `mae_in1k_ft_in1k`, `mae_in1k_ft_plc365`, `mae_in1k_ft_inat18` for `--method`
- Use to `--transfer` and `--freeze` to load and freeze the pre-trained backbone weights.
- Set the number of LLB layer with `--num_nvit_layers` 
- Set the visual feature layer of backbone with `--target_layer` 
- Set the number of latent object of LLB with `--object_size` 

