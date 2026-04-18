**SentiCore — Spiking Multimodal Emotion Recognition**

This repository contains code, data utilities, models, training scripts, and inference pipelines for SentiCore — a spiking / transformer-based multimodal emotion recognition project.

**Quick Start**
- **Install dependencies:** `pip install -r requirements.txt`
- **Prepare data:** place raw files under `data_sorted/` following the existing folder structure (e.g., `data_sorted/angry`, `data_sorted/happy`, ...).
- **Train:** run the training entrypoint: [Train/train_senticore.py](Train/train_senticore.py)
  - Example: `python Train/train_senticore.py`
- **Infer:** run the inference script with the provided config: `python inference/infer.py --config configs/inference_config.json` ([configs/inference_config.json](configs/inference_config.json)).

**Repository Layout (high-level)**
- **`Train/`**: training scripts and utilities. Main training entry: [Train/train_senticore.py](Train/train_senticore.py) which defines `Config` and `Trainer`.
- **`Model/`**: model architectures and wrappers. Key files: [Model/SentiCore_Model.py](Model/SentiCore_Model.py), [Model/spikformer.py](Model/spikformer.py), [Model/Resnet101.py](Model/Resnet101.py), [Model/MLP.py](Model/MLP.py).
- **`dataset/`**: dataset classes, dataloader builders and collate functions. Start with [dataset/build_dataloader.py](dataset/build_dataloader.py) and [dataset/multimodal_dataset.py](dataset/multimodal_dataset.py).
- **`preprocessing/` & `features/`**: scripts that convert raw data into feature representations consumed by datasets.
- **`Loss/`**: custom loss implementations (e.g., [Loss/SoftHGRLoss.py](Loss/SoftHGRLoss.py), [Loss/MultiDSCLoss.py](Loss/MultiDSCLoss.py)).
- **`inference/`**: inference pipeline and utilities. Main pipeline: [inference/pipeline.py](inference/pipeline.py); CLI wrapper: [inference/infer.py](inference/infer.py).
- **`finetuning/`**: hyperparameter tuning and MLflow utilities (e.g., `tune_optuna.py`).
- **`checkpoints/`**: saved experiment checkpoints (organized per experiment).
- **`configs/`**: JSON configs used by inference and utilities.
- **`results.ipynb`**: notebook for result analysis and plotting.

**Core Concepts & Flow**
- **Data flow:** raw data → preprocessing → `features/` → dataset loader (`dataset/*`) → model (`Model/*`) → training (`Train/train_senticore.py`) → checkpoints (`checkpoints/`) → inference (`inference/*`) → results/plots (`results.ipynb`).
- **Config & run naming:** training uses a `Config` class for hyperparameters and generates run names; MLflow is used to track experiments.
- **Checkpoints:** models and optimizer states are saved under `checkpoints/` (experiment-specific subfolders). Use `torch.load` to restore model weights.

**Models (brief)**
- **`SentiCore`** (`Model/SentiCore_Model.py`): the primary multimodal model combining visual/audio/text or feature streams with spiking/temporal components.
- **`Spikformer`** (`Model/spikformer.py`): transformer-like module adapted for spiking inputs and temporal processing.
- **`ResNet101`** (`Model/Resnet101.py`): visual backbone used for extracting image features when applicable.
- **`MLP`** (`Model/MLP.py`): lightweight fully-connected modules for projection/classification heads.

When referencing model classes in code look for `class SentiCore`, `class Spikformer`, and `class ResNet101` inside the files above.

**Losses & Metrics**
- Implementations in `Loss/` include correlation-based and contrastive losses such as SoftHGR (`Loss/SoftHGRLoss.py`), multi-dice/similarity losses, and focal/weighted variants.
- Training logs standard metrics (accuracy, F1, loss components) to MLflow.

**Training**
- Primary entry: [Train/train_senticore.py](Train/train_senticore.py).
- Typical flow inside `Trainer`: dataset creation → model instantiation → optimizer/scheduler setup → training loop with logging and validation → checkpointing via `torch.save`.
- To resume from a checkpoint, load the model state with `trainer.model.load_state_dict(torch.load(<checkpoint_path>))` (see `results.ipynb` example).

**Inference**
- Use [inference/infer.py](inference/infer.py) to run end-to-end inference. The pipeline performs segmentation/feature extraction (if needed) and model inference via [inference/pipeline.py](inference/pipeline.py).
- The inference config is at [configs/inference_config.json](configs/inference_config.json).

**Experiment Tracking & Databases**
- MLflow is integrated (look in `finetuning/` and training scripts). The project uses a SQLite backend by default in some scripts (`sqlite:///snn.db`) for MLflow; check and update URIs before running in a multi-user environment.

**Notebooks & Analysis**
- `results.ipynb` contains example code to load a trained model, collect MLflow metrics from `snn.db`, plot training/validation curves, and print confusion matrices. See the cell that demonstrates loading a checkpoint and running `evaluate_model`.

**Dependencies**
- See `requirements.txt` for full requirements. Critical packages include: `torch`, `mlflow`, `optuna`, `pandas`, `scikit-learn`, `requests`, `beautifulsoup4` (used in helper notebooks/scripts).

**Useful Commands**
- Install deps: `pip install -r requirements.txt`
- Run training: `python Train/train_senticore.py`
- Resume from checkpoint (notebook example):
  ```python
  import torch
  trainer.model.load_state_dict(torch.load(r"checkpoints/<experiment>/checkpoint_epochXX.pt"))
  ```
- Run inference: `python inference/infer.py --config configs/inference_config.json`

**Troubleshooting & Tips**
- Ensure `features/` is prepared before training; preprocessing scripts under `preprocessing/` generate them.
- Update `mlflow.set_tracking_uri(...)` if you prefer a remote MLflow server.
- Check GPU availability before long training runs; `Trainer` expects CUDA when configured.
- Watch for hardcoded paths in notebooks and scripts — replace them with environment-agnostic config values.

**Suggested Next Steps / Improvements**
- Add `CONTRIBUTING.md` with run examples and expected environment setup.
- Add a small smoke-test script to verify imports and device setup.
- Parameterize file paths and MLflow URIs through environment variables or a central config.

If you want, I can also:
- create a short `run_example.bat` that trains one epoch and saves a checkpoint,
- add a smoke-test that imports core modules and runs a dry-forward pass,
- or generate a visual dependency diagram.
