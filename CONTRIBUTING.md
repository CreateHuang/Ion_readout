# Contributing

Thank you for your interest in contributing.

Before opening a pull request:

1. Keep datasets, annotations, checkpoints, generated reports, and figures out of the repository.
2. Use environment variables or command-line arguments for local paths.
3. Run syntax checks before submitting changes:

```bash
python -m py_compile train_main.py Testset_eval.py Ablation_eval.py dataset.py loss.py trainer.py
```
