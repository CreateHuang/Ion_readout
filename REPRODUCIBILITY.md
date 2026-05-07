# Reproducibility Guide

This guide assumes the repository root is the current working directory.

## 1. Environment

Use Python 3.12 with a CUDA-enabled PyTorch installation when GPU training is available.

```bash
pip install torch torchvision
pip install -r requirements.txt
```

If your PyTorch build requires a specific CUDA wheel, install `torch` and `torchvision` from the official PyTorch selector first, then install the remaining dependencies from `requirements.txt`.

## 2. Expected data layout

By default, `config.py` looks for the training dataset at:

```text
./data/intersection_train_data
```

The expected supervised training layout is:

```text
intersection_train_data/
  images/
    *.png
  masks/
    *.png
  site_dia_labels/
    site_coords.npy
    state_hard.npy
    labels_all.npz
    per_sample_npz/
      *.npz
```

You can override the dataset location with environment variables:

```bash
export IMG_DIR=/path/to/intersection_train_data
export SITE_DIA_LABEL_DIR=/path/to/intersection_train_data/site_dia_labels
```

On PowerShell:

```powershell
$env:IMG_DIR="C:\path\to\intersection_train_data"
$env:SITE_DIA_LABEL_DIR="C:\path\to\intersection_train_data\site_dia_labels"
```

## 3. Optional self-supervised pretraining

No pretrained model files are included in this cleaned repository. To create a pretraining checkpoint, run:

```bash
python Pre_train/main_train.py --data_dir /path/to/unlabeled/images --save_dir Pre_train/Run_pretrain
```

After pretraining, pass the checkpoint path to supervised training with `--pretrained_ckpt`.

## 4. Main Site-DIA training

Train Site-DIA without dense mask supervision:

```bash
python train_main.py \
  --model_arch site_dia \
  --pretrained_ckpt Pre_train/Run_pretrain/best.pth \
  --load_mode backbone \
  --site_mask_weight 0 \
  --site_dia_label_dir /path/to/intersection_train_data/site_dia_labels \
  --sample_sizes 50000 \
  --epochs 100 \
  --batch_size 48 \
  --output_dir outputs/site_dia_main
```

To train without a pretrained checkpoint:

```bash
python train_main.py \
  --model_arch site_dia \
  --from_scratch \
  --site_mask_weight 0 \
  --site_dia_label_dir /path/to/intersection_train_data/site_dia_labels \
  --sample_sizes 50000 \
  --epochs 100 \
  --batch_size 48 \
  --output_dir outputs/site_dia_from_scratch
```

## 5. Evaluation

Evaluate a trained checkpoint on a test set:

```bash
python Testset_eval.py \
  --model_arch site_dia \
  --model_path outputs/site_dia_main/sample_50000/best.pth \
  --test_root /path/to/intersection_test_data \
  --gt_all_bright_mask /path/to/GroundTruth.png \
  --result_dir outputs/site_dia_eval
```

## 6. Ablation evaluation

Train each ablation checkpoint separately, then evaluate them together:

```bash
python Ablation_eval.py \
  --full_ckpt outputs/ablation/full/sample_50000/best.pth \
  --no_self_attn_ckpt outputs/ablation/no_self_attn/sample_50000/best.pth \
  --no_psf_ckpt outputs/ablation/no_psf_guided/sample_50000/best.pth \
  --psf_template_only_ckpt outputs/ablation/psf_template_only/sample_50000/best.pth \
  --no_pretrain_ckpt outputs/ablation/no_pretrain/sample_50000/best.pth \
  --test_root /path/to/intersection_test_data \
  --gt_all_bright_mask /path/to/GroundTruth.png \
  --result_dir outputs/ablation_eval
```

## 7. Basic validation

Before running long jobs, check syntax:

```bash
python -m py_compile train_main.py Testset_eval.py Ablation_eval.py dataset.py loss.py trainer.py
```

All generated checkpoints and evaluation outputs will be written under the output directories you specify.

