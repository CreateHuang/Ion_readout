# 300-Ion CNN Detection

This project trains and evaluates neural networks for 300-ion fluorescence image readout. The main implementation supports DW-UNet segmentation models and the Site-DIA readout model, which predicts per-ion bright/dark states from image features and calibrated ion-site geometry.

Model checkpoints, generated outputs, paper figures, remote-server utilities, caches, and experiment logs are intentionally not included in this cleaned project.

## Main files

- `train_main.py`: supervised training entry point.
- `Testset_eval.py`: test-set evaluation entry point.
- `Ablation_eval.py`: evaluation for separately trained ablation checkpoints.
- `dataset.py`, `loss.py`, `trainer.py`: data loading, losses, and training loop utilities.
- `nets/`: model definitions.
- `Pre_train/`: self-supervised denoising pretraining code.
- `requirements.txt`: Python dependencies.
- `REPRODUCIBILITY.md`: commands for reproducing training and evaluation.

See `REPRODUCIBILITY.md` for setup, data layout, training, and evaluation steps.
