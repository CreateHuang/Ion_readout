import argparse

import logging

import os

import random

from pathlib import Path


import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor


from dataset import MaskDataset, get_img_files

from loss import HybridSegmentationMultiIonLoss, SiteDIAMultitaskLoss

from nets.DWNetV2_unet import DWNetV2_unet

from nets.StandardUNet import StandardUNet

from nets.ViTUNet import ViTUNet

from nets.SETR import SETR

from nets.SegFormer import SegFormer

from nets.Segmenter import Segmenter

from trainer import Trainer


torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False


DEFAULT_PRETRAIN_CKPT = os.environ.get("PRETRAIN_CKPT", "")


BATCH_SIZE = 48

NUM_WORKERS = 2

INITIAL_LR = 1e-4

RANDOM_STATE = 1

NUM_EPOCHS = 100

VAL_RATIO = 0.2

SAMPLE_SIZES = [50000]


EXPERIMENT = "pretrained_random_subsets_single_run"

OUT_DIR = Path("outputs") / EXPERIMENT


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)

    random.seed(worker_seed)


def get_data_loaders(

    train_files,

    val_files,

    batch_size,

    site_dia_label_dir=None,

    require_mask=True,

):

    train_transform = Compose(

        [

            ToTensor(),

        ]

    )


    val_transform = Compose(

        [

            ToTensor(),

        ]

    )


    g = torch.Generator()

    g.manual_seed(RANDOM_STATE)


    train_loader = DataLoader(

        MaskDataset(

            train_files,

            transform=train_transform,

            mask_transform=val_transform,

            site_dia_label_dir=site_dia_label_dir,

            require_mask=require_mask,

        ),

        batch_size=batch_size,

        shuffle=True,

        pin_memory=True,

        num_workers=NUM_WORKERS,

        worker_init_fn=seed_worker,

        generator=g,

    )


    val_loader = DataLoader(

        MaskDataset(

            val_files,

            transform=val_transform,

            mask_transform=val_transform,

            site_dia_label_dir=site_dia_label_dir,

            require_mask=require_mask,

        ),

        batch_size=batch_size,

        shuffle=False,

        pin_memory=True,

        num_workers=NUM_WORKERS,

        worker_init_fn=seed_worker,

    )


    return train_loader, val_loader


def build_logger(log_path):

    logger = logging.getLogger("train_subset_logger")

    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    logger.addHandler(logging.FileHandler(filename=log_path, encoding="utf-8"))

    logger.addHandler(logging.StreamHandler())

    return logger


def save_best_model(output_dir, model, df_hist):

    if df_hist["val_loss"].tail(1).iloc[0] <= df_hist["val_loss"].min():

        torch.save(model.state_dict(), output_dir / "best.pth")


def write_on_board(writer, experiment_name, df_hist):

    row = df_hist.tail(1).iloc[0]


    writer.add_scalars(

        f"{experiment_name}/loss",

        {

            "train": row.train_loss,

            "val": row.val_loss,

        },

        row.epoch,

    )


def log_hist(logger, df_hist):

    last = df_hist.tail(1)

    best = df_hist.sort_values("val_loss").head(1)

    summary = pd.concat((last, best)).reset_index(drop=True)

    summary["name"] = ["Last", "Best"]

    logger.debug(summary[["name", "epoch", "train_loss", "val_loss", "current_lr"]])

    logger.debug("")


def save_loss_plot(df_hist, output_path, plot_title):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", linewidth=2)

    ax.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Loss")

    ax.set_title(plot_title)

    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend()

    fig.tight_layout()

    fig.savefig(output_path, dpi=200)

    plt.close(fig)


def load_pretrained_weights(model, ckpt_path, device, load_mode="full"):

    if not ckpt_path:

        print("No pretrained checkpoint provided.")

        return model


    if not os.path.exists(ckpt_path):

        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")


    ckpt = torch.load(ckpt_path, map_location=device)


    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:

        state_dict = ckpt["model_state_dict"]

    elif isinstance(ckpt, dict) and "state_dict" in ckpt:

        state_dict = ckpt["state_dict"]

    else:

        state_dict = ckpt


    stripped = {}

    for k, v in state_dict.items():

        new_k = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k

        stripped[new_k] = v

    state_dict = stripped


    model_state = model.state_dict()

    loaded_keys = []


    if load_mode == "full":

        filtered = {}

        for k, v in state_dict.items():

            if k in model_state and model_state[k].shape == v.shape:

                filtered[k] = v

                loaded_keys.append(k)


        incompatible = model.load_state_dict(filtered, strict=False)

        print(f"[Pretrain-FULL] loaded={len(loaded_keys)}")

        print(f"[Pretrain-FULL] missing={len(incompatible.missing_keys)}")

        print(f"[Pretrain-FULL] unexpected={len(incompatible.unexpected_keys)}")


    elif load_mode == "backbone":

        filtered = {}

        for k, v in state_dict.items():

            candidate_keys = [k]


            if k.startswith("backbone."):

                candidate_keys.append(k[len("backbone."):])


            for ck in candidate_keys:

                if ck in model_state and model_state[ck].shape == v.shape:

                    filtered[ck] = v

                    loaded_keys.append(ck)

                    break


                bk = f"backbone.{ck}"

                if bk in model_state and model_state[bk].shape == v.shape:

                    filtered[bk] = v

                    loaded_keys.append(bk)

                    break


        incompatible = model.load_state_dict(filtered, strict=False)

        print(f"[Pretrain-BACKBONE] loaded={len(loaded_keys)}")

        print(f"[Pretrain-BACKBONE] missing={len(incompatible.missing_keys)}")

        print(f"[Pretrain-BACKBONE] unexpected={len(incompatible.unexpected_keys)}")


    else:

        raise ValueError(f"Unsupported load_mode: {load_mode}")


    return model


def select_subset(image_files, sample_size, seed):

    if sample_size > len(image_files):

        raise ValueError(

            f"Requested sample size {sample_size}, but only {len(image_files)} images are available."

        )


    rng = np.random.default_rng(seed)

    selected_indices = np.sort(

        rng.choice(len(image_files), size=sample_size, replace=False)

    )

    return image_files[selected_indices]


def split_train_val(subset_files, val_ratio, seed):

    train_files, val_files = train_test_split(

        subset_files,

        test_size=val_ratio,

        random_state=seed,

        shuffle=True,

    )

    return np.array(train_files), np.array(val_files)


def save_split_manifest(train_files, val_files, output_path):

    split_df = pd.DataFrame(

        {

            "file": list(train_files) + list(val_files),

            "split": ["train"] * len(train_files) + ["val"] * len(val_files),

        }

    )

    split_df.to_csv(output_path, index=False)


def write_experiment_readme(

    output_dir, dataset_dir, pretrained_ckpt, load_mode, batch_size, use_amp

):

    readme_text = f"""Pretrained Training on Random Subsets

This directory stores pretrained training runs on random subsets of the dataset.

Dataset

- Dataset root: `{dataset_dir}`
- Image counts tested: {", ".join(str(size) for size in SAMPLE_SIZES)}
- Validation split ratio: {VAL_RATIO}
- Max epochs per run: {NUM_EPOCHS}
- Batch size: {batch_size}
- AMP mixed precision: {use_amp}

Pretrained Model

- Checkpoint: `{pretrained_ckpt}`
- Load mode: `{load_mode}`

Output Structure

Each `sample_xxxx` directory contains:

- `best.pth` for the best validation-loss checkpoint
- `hist.csv` with per-epoch losses and learning rate
- `loss.png` with the train/validation loss curve
- `split_manifest.csv` listing the train/validation files
- `tensorboard` logs
- `summary.csv` with the key results for that sample-size run

Run Command

```bash
python train_main.py --pretrained_ckpt "{pretrained_ckpt}" --load_mode {load_mode}
```
"""

    (output_dir / "README.md").write_text(readme_text, encoding="utf-8")


def run_experiments(

    pre_trained,

    pretrained_ckpt="",

    load_mode="full",

    batch_size=BATCH_SIZE,

    use_amp=True,

    centroid_weight=0.0,

    bce_weight=1.0,

    dice_weight=1.0,

    model_arch="dwunet",

    output_dir=OUT_DIR,

    site_dia_label_dir="",

    site_mask_weight=0.2,

    site_state_weight=1.0,

    site_coord_weight=0.05,

    site_exist_weight=0.1,

    site_offset_reg_weight=0.01,

    freeze_dwunet=False,

    dia_num_ion_attn_layers=1,

    dia_use_psf_guided_offsets=True,

    dia_residual_attn_offset=1.0,

    early_stopping_patience=20,

):

    require_mask = not (model_arch == "site_dia" and site_mask_weight == 0)

    image_files = get_img_files(require_masks=require_mask)

    if len(image_files) == 0:

        raise RuntimeError(

            "No training images found. Please check IMG_DIR/images and IMG_DIR/masks."

        )


    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(output_dir / "run.log")


    device = torch.device("cuda")

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info(f"CUDA: {torch.version.cuda}")


    write_experiment_readme(

        output_dir=output_dir,

        dataset_dir=os.environ.get(

            "IMG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "intersection_train_data")

        ),

        pretrained_ckpt=pretrained_ckpt,

        load_mode=load_mode,

        batch_size=batch_size,

        use_amp=bool(use_amp),

    )


    logger.info(f"Found {len(image_files)} available image/mask pairs.")

    logger.info(f"Model architecture: {model_arch}")

    logger.info(f"Batch size: {batch_size}")

    logger.info(f"AMP mixed precision: {bool(use_amp)}")

    all_summaries = []


    for sample_size in SAMPLE_SIZES:

        logger.info("=" * 80)

        logger.info(f"Starting sample-size experiment: {sample_size}")


        sample_dir = output_dir / f"sample_{sample_size}"

        sample_dir.mkdir(parents=True, exist_ok=True)


        subset_files = select_subset(

            image_files, sample_size, seed=RANDOM_STATE + sample_size

        )

        train_files, val_files = split_train_val(

            subset_files,

            val_ratio=VAL_RATIO,

            seed=RANDOM_STATE + sample_size,

        )

        save_split_manifest(train_files, val_files, sample_dir / "split_manifest.csv")


        writer = SummaryWriter(log_dir=str(sample_dir / "tensorboard"))

        experiment_name = f"{EXPERIMENT}/sample_{sample_size}"


        def on_after_epoch(

            m,

            df_hist,

            current_sample_dir=sample_dir,

            current_experiment_name=experiment_name,

        ):

            save_best_model(current_sample_dir, m, df_hist)

            write_on_board(writer, current_experiment_name, df_hist)

            log_hist(logger, df_hist)


        if model_arch == "site_dia":

            if not site_dia_label_dir:

                site_dia_label_dir = os.environ.get("SITE_DIA_LABEL_DIR", "")

            if not site_dia_label_dir:

                raise ValueError(

                    "site_dia requires --site_dia_label_dir or SITE_DIA_LABEL_DIR."

                )

            criterion = SiteDIAMultitaskLoss(

                mask_weight=site_mask_weight,

                state_weight=site_state_weight,

                coord_weight=site_coord_weight,

                exist_weight=site_exist_weight,

                offset_reg_weight=site_offset_reg_weight,

                bce_weight=bce_weight,

                dice_weight=dice_weight,

            )

        else:

            criterion = HybridSegmentationMultiIonLoss(

                bce_weight=bce_weight,

                dice_weight=dice_weight,

                centroid_weight=centroid_weight,

                radius=4,

            )


        data_loaders = get_data_loaders(

            train_files,

            val_files,

            batch_size=batch_size,

            site_dia_label_dir=site_dia_label_dir if model_arch == "site_dia" else None,

            require_mask=require_mask,

        )


        if model_arch == "dwunet":

            model = DWNetV2_unet(pre_trained)

        elif model_arch == "site_dia":

            model = DWNetV2_unet(

                pre_trained,

                enable_site_dia=True,

                num_ions=300,

                dia_num_ion_attn_layers=dia_num_ion_attn_layers,

                dia_use_psf_guided_offsets=dia_use_psf_guided_offsets,

                dia_residual_attn_offset=dia_residual_attn_offset,

            )

        elif model_arch == "standard_unet":

            model = StandardUNet()

            if pretrained_ckpt:

                logger.warning(

                    "standard_unet does not use DW-UNet pretraining; training from scratch."

                )

                pretrained_ckpt = ""

        elif model_arch == "vit_unet":

            model = ViTUNet()

            if pretrained_ckpt:

                logger.warning("vit_unet does not use DW-UNet pretraining; training from scratch.")

                pretrained_ckpt = ""

        elif model_arch == "setr":

            model = SETR()

            if pretrained_ckpt:

                logger.warning("setr does not use DW-UNet pretraining; training from scratch.")

                pretrained_ckpt = ""

        elif model_arch == "segformer":

            model = SegFormer()

            if pretrained_ckpt:

                logger.warning("segformer does not use DW-UNet pretraining; training from scratch.")

                pretrained_ckpt = ""

        elif model_arch == "segmenter":

            model = Segmenter()

            if pretrained_ckpt:

                logger.warning("segmenter does not use DW-UNet pretraining; training from scratch.")

                pretrained_ckpt = ""

        else:

            raise ValueError(f"Unsupported model_arch: {model_arch}")


        if model_arch == "site_dia" and freeze_dwunet:

            for name, param in model.named_parameters():

                param.requires_grad = name.startswith("site_dia.")

            logger.info("Frozen DW-UNet backbone/decoder; training Site-DIA head only.")


        model.to(device)


        if pretrained_ckpt:

            logger.info(f"Loading pretrained checkpoint: {pretrained_ckpt}")

            logger.info(f"Load mode: {load_mode}")

            model = load_pretrained_weights(

                model=model,

                ckpt_path=pretrained_ckpt,

                device=device,

                load_mode=load_mode,

            )


        optimizer = torch.optim.Adam(

            [p for p in model.parameters() if p.requires_grad], lr=INITIAL_LR, weight_decay=1e-5

        )


        scheduler = ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.75,

            patience=3,

            threshold=1e-4,

            min_lr=1e-6,

        )


        trainer = Trainer(

            data_loaders=data_loaders,

            criterion=criterion,

            device=device,

            scheduler=scheduler,

            on_after_epoch=on_after_epoch,

            use_amp=use_amp,

            early_stopping_patience=early_stopping_patience,

        )


        hist = trainer.train(model, optimizer, num_epochs=NUM_EPOCHS)

        hist.to_csv(sample_dir / "hist.csv", index=False)

        save_loss_plot(

            hist,

            output_path=sample_dir / "loss.png",

            plot_title=f"Sample {sample_size} Loss Curve",

        )


        best_row = hist.loc[hist["val_loss"].idxmin()]

        last_row = hist.iloc[-1]

        summary_df = pd.DataFrame(

            [

                {

                    "sample_size": sample_size,

                    "num_epochs": NUM_EPOCHS,

                    "train_size": len(train_files),

                    "val_size": len(val_files),

                    "best_epoch": int(best_row["epoch"]),

                    "best_val_loss": float(best_row["val_loss"]),

                    "last_val_loss": float(last_row["val_loss"]),

                    "last_lr": float(last_row["current_lr"]),

                }

            ]

        )

        summary_df.to_csv(sample_dir / "summary.csv", index=False)

        all_summaries.append(summary_df.iloc[0].to_dict())


        writer.close()

        del (

            trainer,

            optimizer,

            scheduler,

            model,

            data_loaders,

            criterion,

            hist,

            summary_df,

        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    pd.DataFrame(all_summaries).to_csv(

        output_dir / "all_results_summary.csv", index=False

    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--pretrained_ckpt",

        type=str,

        default=DEFAULT_PRETRAIN_CKPT,

        help="Path to pretrained checkpoint",

    )

    parser.add_argument(

        "--load_mode", type=str, default="full", choices=["full", "backbone"]

    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default=BATCH_SIZE,

        help="Batch size for both training and validation",

    )

    parser.add_argument(

        "--disable_amp",

        action="store_true",

        help="Disable CUDA mixed precision training",

    )

    parser.add_argument(

        "--from_scratch",

        action="store_true",

        help="Do not load a self-supervised pretrained checkpoint",

    )

    parser.add_argument(

        "--centroid_weight",

        type=float,

        default=0.0,

        help="Weight for the multi-ion centroid/localization loss ablation",

    )

    parser.add_argument(

        "--bce_weight",

        type=float,

        default=1.0,

        help="Weight for BCEWithLogits loss; set to 0 for Dice-only training",

    )

    parser.add_argument(

        "--dice_weight",

        type=float,

        default=1.0,

        help="Weight for Dice loss",

    )

    parser.add_argument(

        "--sample_sizes",

        type=str,

        default=",".join(str(x) for x in SAMPLE_SIZES),

        help="Comma-separated subset sizes, e.g. 1000,2000,5000",

    )

    parser.add_argument(

        "--epochs",

        type=int,

        default=NUM_EPOCHS,

        help="Number of epochs per subset experiment",

    )

    parser.add_argument(

        "--model_arch",

        type=str,

        default="dwunet",

        choices=["dwunet", "standard_unet", "site_dia", "vit_unet", "setr", "segformer", "segmenter"],

        help="Architecture to train for architecture ablations",

    )

    parser.add_argument(

        "--output_dir",

        type=str,

        default=str(OUT_DIR),

        help="Output directory for this experiment; use separate dirs for ablations",

    )

    parser.add_argument(

        "--site_dia_label_dir",

        type=str,

        default=os.environ.get("SITE_DIA_LABEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "intersection_train_data", "site_dia_labels")),

        help="Directory containing Site-DIA site-level center/state labels",

    )

    parser.add_argument("--site_mask_weight", type=float, default=0.2)

    parser.add_argument("--site_state_weight", type=float, default=1.0)

    parser.add_argument("--site_coord_weight", type=float, default=0.05)

    parser.add_argument("--site_exist_weight", type=float, default=0.1)

    parser.add_argument("--site_offset_reg_weight", type=float, default=0.01)

    parser.add_argument(

        "--freeze_dwunet",

        action="store_true",

        help="For Site-DIA stage-2 training: freeze the DW-UNet backbone/decoder and train only the Site-DIA head.",

    )

    parser.add_argument(

        "--dia_num_ion_attn_layers",

        type=int,

        default=1,

        help="Number of ion-ion self-attention layers for PSF-crosstalk suppression (0 to disable).",

    )

    parser.add_argument(

        "--disable_psf_guided_offsets",

        action="store_true",

        help="Ablation training: use generic learned deformable offsets instead of nearest-neighbor PSF-guided offsets.",

    )

    parser.add_argument(

        "--dia_residual_attn_offset",

        type=float,

        default=1.0,

        help="Maximum learned residual offset around the PSF template; set 0 for PSF-template-only ablation.",

    )

    parser.add_argument(

        "--random_state",

        type=int,

        default=RANDOM_STATE,

        help="Global random seed for reproducibility (use 1/2/3 for multi-seed runs).",

    )

    parser.add_argument(

        "--early_stopping_patience",

        type=int,

        default=20,

        help="Stop training if val_loss does not improve for this many epochs (default: 20).",

    )

    args = parser.parse_args()


    RANDOM_STATE = args.random_state

    random.seed(RANDOM_STATE)

    np.random.seed(RANDOM_STATE)

    torch.manual_seed(RANDOM_STATE)

    torch.cuda.manual_seed_all(RANDOM_STATE)


    SAMPLE_SIZES[:] = [

        int(x.strip()) for x in args.sample_sizes.split(",") if x.strip()

    ]

    NUM_EPOCHS = args.epochs

    if args.from_scratch:

        args.pretrained_ckpt = ""


    if torch.cuda.is_available():

        torch.cuda.reset_max_memory_allocated()

        torch.cuda.reset_accumulated_memory_stats()


    run_experiments(

        pre_trained=None,

        pretrained_ckpt=args.pretrained_ckpt,

        load_mode=args.load_mode,

        batch_size=args.batch_size,

        use_amp=not args.disable_amp,

        centroid_weight=args.centroid_weight,

        bce_weight=args.bce_weight,

        dice_weight=args.dice_weight,

        model_arch=args.model_arch,

        output_dir=args.output_dir,

        site_dia_label_dir=args.site_dia_label_dir,

        site_mask_weight=args.site_mask_weight,

        site_state_weight=args.site_state_weight,

        site_coord_weight=args.site_coord_weight,

        site_exist_weight=args.site_exist_weight,

        site_offset_reg_weight=args.site_offset_reg_weight,

        freeze_dwunet=args.freeze_dwunet,

        dia_num_ion_attn_layers=args.dia_num_ion_attn_layers,

        dia_use_psf_guided_offsets=not args.disable_psf_guided_offsets,

        dia_residual_attn_offset=args.dia_residual_attn_offset,

        early_stopping_patience=args.early_stopping_patience,

    )

