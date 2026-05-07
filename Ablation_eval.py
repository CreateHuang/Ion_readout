import argparse

import os

import time

from pathlib import Path


import cv2

import numpy as np

import torch


from nets.DWNetV2_unet import DWNetV2_unet

from Testset_eval import (

    DEFAULT_GT_ALL_BRIGHT_MASK,

    DEFAULT_MODEL_PATH,

    DEFAULT_TEST_ROOT,

    MASK_THRESHOLD,

    PRED_THRESHOLD,

    binary_average_precision,

    binary_auroc,

    calc_binary_metrics,

    collect_pairs,

    expected_calibration_error,

    extract_fixed_sites_from_gt,

    format_value,

    load_checkpoint_state,

    load_gray_image,

    make_state_labels_from_mask,

    make_table,

    preprocess_input,

    safe_nll,

)


DEFAULT_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "site_dia_ablation_result")


def ensure_dir(path):

    os.makedirs(path, exist_ok=True)


def instantiate_model(cfg, device):

    model = DWNetV2_unet(

        pre_trained=None,

        mode="eval",

        enable_site_dia=True,

        num_ions=300,

        dia_num_ion_attn_layers=cfg["num_ion_attn_layers"],

        dia_use_psf_guided_offsets=cfg["use_psf_guided_offsets"],

        dia_residual_attn_offset=cfg["residual_attn_offset"],

        dia_psf_sigma=cfg["psf_sigma"],

    )

    state_dict = load_checkpoint_state(cfg["checkpoint"], device)

    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=cfg["strict_load"])

    model.to(device)

    model.eval()

    return model, incompatible


def evaluate_variant(model, pairs, coords, coords_int, device, args):

    coords_t = torch.from_numpy(coords).float().unsqueeze(0).to(device)


    y_true_all = []

    y_prob_all = []

    y_pred_all = []

    uncertainty_all = []

    coord_l2_all = []

    count_abs_errors = []

    exact_count_matches = []

    bitstring_matches = []

    mask_dice = []

    mask_iou = []

    mask_f1 = []

    latency_ms = []

    true_counts = []


    with torch.no_grad():

        for idx, (img_path, mask_path) in enumerate(pairs, start=1):

            gray = load_gray_image(img_path)

            mask_img = load_gray_image(mask_path)

            gt_bin = (mask_img > MASK_THRESHOLD).astype(np.uint8)

            state_true = make_state_labels_from_mask(mask_img, coords_int)

            true_count = int(state_true.sum())

            true_counts.append(true_count)


            inp = preprocess_input(gray).to(device)

            if device.type == "cuda":

                torch.cuda.synchronize()

            t0 = time.perf_counter()

            out = model(inp, site_coords=coords_t)

            if device.type == "cuda":

                torch.cuda.synchronize()

            latency_ms.append((time.perf_counter() - t0) * 1000.0)


            state_prob = torch.sigmoid(out["bright_logit"]).squeeze(0).detach().cpu().numpy()

            state_pred = (state_prob >= args.state_threshold).astype(np.uint8)

            pred_count = int(state_pred.sum())

            count_abs_errors.append(abs(pred_count - true_count))

            exact_count_matches.append(int(pred_count == true_count))

            bitstring_matches.append(int(np.all(state_pred == state_true)))


            pred_coords = out["pred_coords"].squeeze(0).detach().cpu().numpy()

            coord_l2_all.append(np.sqrt(((pred_coords - coords) ** 2).sum(axis=1)))


            if "uncertainty" in out:

                uncertainty_all.append(out["uncertainty"].squeeze(0).detach().cpu().numpy())


            mask_prob = torch.sigmoid(out["mask_logits"]).squeeze().detach().cpu().numpy()

            pred_bin = (mask_prob >= args.mask_threshold).astype(np.uint8)

            mm = calc_binary_metrics(pred_bin, gt_bin)

            mask_dice.append(mm["dice"])

            mask_iou.append(mm["iou"])

            mask_f1.append(mm["f1"])


            y_true_all.append(state_true)

            y_prob_all.append(state_prob)

            y_pred_all.append(state_pred)


            if idx % args.log_every == 0 or idx == len(pairs):

                print(f"  [{idx}/{len(pairs)}] {img_path.name}")


    y_true = np.concatenate(y_true_all)

    y_prob = np.concatenate(y_prob_all)

    y_pred = np.concatenate(y_pred_all)

    state_metrics = calc_binary_metrics(y_pred, y_true)

    dark_metrics = calc_binary_metrics(1 - y_pred, 1 - y_true)

    coord_l2 = np.concatenate(coord_l2_all)

    latency_ms = np.asarray(latency_ms, dtype=np.float64)

    count_abs_errors = np.asarray(count_abs_errors, dtype=np.float64)

    uncertainty = np.concatenate(uncertainty_all) if uncertainty_all else np.asarray([], dtype=np.float64)


    return {

        "per_ion_acc": state_metrics["accuracy"],

        "bright_f1": state_metrics["f1"],

        "bright_precision": state_metrics["precision"],

        "bright_recall": state_metrics["recall"],

        "dark_recall": dark_metrics["recall"],

        "balanced_acc": 0.5 * (state_metrics["recall"] + state_metrics["specificity"]),

        "bitstring_exact": float(np.mean(bitstring_matches)),

        "count_mae": float(count_abs_errors.mean()),

        "exact_count_match": float(np.mean(exact_count_matches)),

        "state_count_fidelity": 1.0 - float(count_abs_errors.sum()) / max(1.0, float(np.sum(true_counts))),

        "mask_dice": float(np.mean(mask_dice)),

        "mask_iou": float(np.mean(mask_iou)),

        "mask_f1": float(np.mean(mask_f1)),

        "auroc": binary_auroc(y_true, y_prob),

        "auprc": binary_average_precision(y_true, y_prob),

        "brier": float(np.mean((y_prob - y_true) ** 2)),

        "nll": safe_nll(y_true, y_prob),

        "ece": expected_calibration_error(y_true, y_prob, n_bins=args.ece_bins),

        "coord_l2_mean": float(coord_l2.mean()),

        "coord_l2_p95": float(np.percentile(coord_l2, 95)),

        "latency_mean_ms": float(latency_ms.mean()),

        "latency_p95_ms": float(np.percentile(latency_ms, 95)),

        "uncertainty_mean": float(uncertainty.mean()) if uncertainty.size else float("nan"),

    }


def build_variant_configs(args):


    required = {

        "--full_ckpt": args.full_ckpt,

        "--no_self_attn_ckpt": args.no_self_attn_ckpt,

        "--no_psf_ckpt": args.no_psf_ckpt,

        "--psf_template_only_ckpt": args.psf_template_only_ckpt,

        "--no_pretrain_ckpt": args.no_pretrain_ckpt,

    }

    missing = [name for name, value in required.items() if not value]

    if missing:

        raise ValueError(

            "Formal retrained ablation requires separate checkpoints for every variant. "

            f"Missing arguments: {', '.join(missing)}.\n"

            "Train each variant first with train_retrained_ablation.py or the commands in "

            "Reproducibility.txt. Do not reuse the full checkpoint for ablation rows."

        )


    rows = [

        {

            "name": "Full no-mask Site-DIA",

            "checkpoint": args.full_ckpt,

            "strict_load": True,

            "use_psf_guided_offsets": True,

            "num_ion_attn_layers": args.full_ion_attn_layers,

            "residual_attn_offset": 1.0,

            "psf_sigma": 1.5,

            "description": "Retrained full Site-DIA model: physics pretrain + PSF-guided offsets + ion-token self-attention.",

        },

        {

            "name": "w/o ion self-attn",

            "checkpoint": args.no_self_attn_ckpt,

            "strict_load": True,

            "use_psf_guided_offsets": True,

            "num_ion_attn_layers": 0,

            "residual_attn_offset": 1.0,

            "psf_sigma": 1.5,

            "description": "Retrained without token-token self-attention from the start of training.",

        },

        {

            "name": "w/o PSF-guided offsets",

            "checkpoint": args.no_psf_ckpt,

            "strict_load": True,

            "use_psf_guided_offsets": False,

            "num_ion_attn_layers": args.full_ion_attn_layers,

            "residual_attn_offset": 4.0,

            "psf_sigma": 1.5,

            "description": "Retrained with generic learned deformable offsets instead of nearest-neighbor PSF-guided sampling.",

        },

        {

            "name": "PSF template only",

            "checkpoint": args.psf_template_only_ckpt,

            "strict_load": True,

            "use_psf_guided_offsets": True,

            "num_ion_attn_layers": args.full_ion_attn_layers,

            "residual_attn_offset": 0.0,

            "psf_sigma": 1.5,

            "description": "Retrained with learned residual offsets disabled; sampling uses only the nearest-neighbor PSF template.",

        },

        {

            "name": "w/o phys. pretraining",

            "checkpoint": args.no_pretrain_ckpt,

            "strict_load": True,

            "use_psf_guided_offsets": True,

            "num_ion_attn_layers": args.full_ion_attn_layers,

            "residual_attn_offset": 1.0,

            "psf_sigma": 1.5,

            "description": "Retrained from scratch without physics-aware self-supervised initialization.",

        },

    ]


    not_found = [(row["name"], row["checkpoint"]) for row in rows if not os.path.exists(row["checkpoint"])]

    if not_found:

        details = "\n".join(f"  - {name}: {path}" for name, path in not_found)

        raise FileNotFoundError(

            "Some retrained ablation checkpoints do not exist:\n"

            f"{details}\n"

            "Run train_retrained_ablation.py first or pass the correct checkpoint paths."

        )

    return rows


def render_ablation_table(results):

    columns = [

        ("Variant", "name"),

        ("Per-ion Acc.", "per_ion_acc"),

        ("Bright F1", "bright_f1"),

        ("Dark Recall", "dark_recall"),

        ("Bitstring", "bitstring_exact"),

        ("Count MAE", "count_mae"),

        ("Mask Dice", "mask_dice"),

        ("ECE", "ece"),

        ("Coord L2", "coord_l2_mean"),

        ("Latency ms", "latency_mean_ms"),

    ]

    header = [c[0] for c in columns]

    table_rows = []

    for r in results:

        table_rows.append([r["name"]] + [format_value(r.get(k)) for _, k in columns[1:]])

    widths = [len(h) for h in header]

    for row in table_rows:

        for i, cell in enumerate(row):

            widths[i] = max(widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    lines = [sep]

    lines.append("| " + " | ".join(header[i].ljust(widths[i]) for i in range(len(header))) + " |")

    lines.append(sep)

    for row in table_rows:

        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(header))) + " |")

    lines.append(sep)

    return "\n".join(lines)


def main():

    args = parse_args()

    ensure_dir(args.result_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    pairs = collect_pairs(args.test_root)

    if args.max_samples > 0:

        pairs = pairs[: args.max_samples]

    coords, coords_int, _, _ = extract_fixed_sites_from_gt(args.gt_all_bright_mask)


    variants = build_variant_configs(args)

    results = []

    for cfg in variants:

        if not os.path.exists(cfg["checkpoint"]):

            print(f"Skipping {cfg['name']}: checkpoint not found: {cfg['checkpoint']}")

            continue

        print("=" * 80)

        print(f"Evaluating {cfg['name']}")

        print(f"Checkpoint: {cfg['checkpoint']}")

        model, incompatible = instantiate_model(cfg, device)

        metrics = evaluate_variant(model, pairs, coords, coords_int, device, args)

        metrics.update(

            {

                "name": cfg["name"],

                "checkpoint": cfg["checkpoint"],

                "description": cfg["description"],

                "missing_keys": len(getattr(incompatible, "missing_keys", [])),

                "unexpected_keys": len(getattr(incompatible, "unexpected_keys", [])),

            }

        )

        results.append(metrics)

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    if not results:

        raise RuntimeError("No ablation variant was evaluated. Check checkpoint paths.")


    table = render_ablation_table(results)

    detail_rows = []

    for r in results:

        detail_rows.extend(

            [

                (f"{r['name']} checkpoint", r["checkpoint"], r["description"]),

                (f"{r['name']} AUROC", r["auroc"], "Threshold-free bright/dark ranking metric."),

                (f"{r['name']} AUPRC", r["auprc"], "Average precision for bright-state prediction."),

                (f"{r['name']} NLL", r["nll"], "Negative log-likelihood; lower is better."),

                (f"{r['name']} Brier", r["brier"], "Probability mean-squared error; lower is better."),

                (f"{r['name']} Count fidelity", r["state_count_fidelity"], "1 - total absolute count error / total true bright count."),

                (f"{r['name']} load missing/unexpected", f"{r['missing_keys']}/{r['unexpected_keys']}", "Nonzero values mean partial-load diagnostic ablation rather than a separately trained checkpoint."),

            ]

        )


    detail_table = make_table(detail_rows)

    report = f"""
Site-DIA ablation study
=======================
Test root        : {args.test_root}
GT site mask     : {args.gt_all_bright_mask}
Evaluated samples: {len(pairs)}
Output file      : {Path(args.result_dir) / args.report_name}

Main ablation table
-------------------
{table}

Detailed metrics and notes
--------------------------
{detail_table}

Explanation
-----------
This ablation isolates the main Site-DIA innovations:
1. Ion-token self-attention tests whether token-token communication helps suppress PSF-tail crosstalk between neighboring ions.
2. PSF-guided offsets test whether nearest-neighbor physics-guided sampling is better than generic deformable sampling.
3. PSF template only tests whether learned residual offsets around the PSF template are necessary.
4. w/o phys. pretraining tests whether physics-aware self-supervised pretraining improves readout accuracy vs. training Site-DIA from scratch.

This table is a formal retrained ablation: each row is loaded from its own separately trained checkpoint. No row reuses the full checkpoint with modules switched off only at inference time.
"""

    out_path = Path(args.result_dir) / args.report_name

    with open(out_path, "w", encoding="utf-8") as f:

        f.write(report)

    print("\nAblation finished.")

    print(f"Report: {out_path}")

    print(table)


def parse_args():

    parser = argparse.ArgumentParser(description="Ablation evaluation for Site-DIA on intersection_test_data.")

    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Deprecated; use --full_ckpt for formal retrained ablation.")

    parser.add_argument("--full_ckpt", type=str, required=True, help="Separately trained checkpoint for the full no-mask Site-DIA model.")

    parser.add_argument("--no_self_attn_ckpt", type=str, required=True, help="Separately trained checkpoint without ion self-attention.")

    parser.add_argument("--no_psf_ckpt", type=str, required=True, help="Separately trained checkpoint without PSF-guided offsets.")

    parser.add_argument("--psf_template_only_ckpt", type=str, required=True, help="Separately trained checkpoint for PSF-template-only offset ablation.")

    parser.add_argument("--no_pretrain_ckpt", type=str, required=True, help="Separately trained checkpoint without physics-aware self-supervised pretraining.")

    parser.add_argument("--test_root", type=str, default=DEFAULT_TEST_ROOT)

    parser.add_argument("--gt_all_bright_mask", type=str, default=DEFAULT_GT_ALL_BRIGHT_MASK)

    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR)

    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto")

    parser.add_argument("--report_name", type=str, default="ablation.txt")

    parser.add_argument("--mask_threshold", type=float, default=PRED_THRESHOLD)

    parser.add_argument("--state_threshold", type=float, default=PRED_THRESHOLD)

    parser.add_argument("--ece_bins", type=int, default=15)

    parser.add_argument("--full_ion_attn_layers", type=int, default=1)

    parser.add_argument("--strict_load", action="store_true", help="Strictly load full checkpoints. Default is non-strict for convenience.")

    parser.add_argument("--strict_load_if_ckpt_provided", action="store_true", help="Use strict loading when a separate variant checkpoint is provided.")

    parser.add_argument("--max_samples", type=int, default=0, help="Debug option: evaluate only first N samples if >0.")

    parser.add_argument("--log_every", type=int, default=200)

    return parser.parse_args()


if __name__ == "__main__":

    main()

