# SNN-Research — Multimodal Spiking Emotion Recognition

This repository contains code for training and running a multimodal emotion recognition system that uses a spiking transformer (Spikformer) together with multimodal fusion. The model combines text, audio, and visual features to classify short spoken utterances into emotions.

This README explains the project purpose, the overall flow, and what each major file and folder does. The language is kept simple so it's easy to follow.

**Purpose**
- Build and train a multimodal emotion classifier that reads video (audio + frames) and transcribed text.
- Use spiking neural network ideas (Spikformer) to modulate features before multimodal fusion.
- Provide a production-friendly inference pipeline that segments videos and returns one label per detected utterance.

**High-level overview (simple)**
1. Segment videos into short spoken utterances using Whisper (phase 1).
2. For each utterance, extract features: text (RoBERTa), audio (Wav2Vec2), and visual (ResNet/ResNet50 features) (phase 2).
3. Save features to disk (`features/train` and `features/validation`).
4. Train a model that projects each modality into a common space, applies a Spikformer to compute spiking modulation, fuses modalities with a Multi-Attention block, and classifies with an MLP.
5. Log experiments with MLflow; optionally tune hyperparameters with Optuna.
6. Run inference on new videos with a lightweight pipeline that mirrors preprocessing and loads saved weights.

**File / Folder Structure (top-level, trimmed)**
- `final/` — Main project code (training, preprocessing, models, losses, finetuning, inference)
  - `Train/` — Training utilities and scripts (e.g. `train_senticore.py`, `save_config.py`)
  - `preprocessing/` — Phase-1 (segmentation) and Phase-2 (feature extraction) code and helpers
  - `dataset/` — Dataloader, dataset class, collate function
  - `Model/` — Model definitions: `spikformer.py`, `SentiCore_Model.py`, `MultiAttn.py`, `MLP.py`, visual backbone
  - `Loss/` — Custom loss implementations (SoftHGR, MultiDSC, etc.)
  - `finetuning/` — Optuna tuning and MLflow helpers
  - `inference/` — Production inference pipeline and small copies of preprocessing/encoders used for deployment
  - `features/` — Output of phase-2 feature extraction (created at runtime): `train/` and `validation/` with `text.pkl`, `audio.pkl`, `visual.pkl`
  - `checkpoints/` — Saved model checkpoints created during training

**Detailed component explanations**

1) Phase‑1: Segmentation
- Purpose: Split long videos into short utterances that contain a single spoken turn.
- Main code: `preprocessing/phase1_segmentation/whisper_segmenter.py`, `build_segments.py`.
- How it works: The Whisper model transcribes and gives word timestamps; we group timestamps into utterances and filter out very short or too-long segments.

2) Phase‑2: Feature Extraction
- Purpose: Convert each short utterance into three fixed-size numeric features: text, audio, visual.
- Main code: `preprocessing/phase2_features/build_features.py` plus `text_features.py`, `audio_features.py`, `visual_features.py`, `video_decoder.py`.
- Text: encoder uses RoBERTa to produce a 768-d embedding (average over tokens).
- Audio: encoder uses Wav2Vec2 to produce a 768-d embedding (pooled mean over time).
- Visual: backbone is a ResNet-based CNN to produce a 2048-d embedding per utterance (pooled across frames).
- Output: For each split (`train`, `validation`) the builder writes three pickles: `text.pkl`, `audio.pkl`, `visual.pkl`. Each file is a dict with keys `features` (Tensor) and `labels` (Tensor).

3) Dataset & DataLoader
- Purpose: Load the pickled features and present them to the model in minibatches.
- Files: `dataset/multimodal_dataset.py`, `dataset/collate.py`, `dataset/build_dataloader.py`.
- The collate function stacks the modality tensors and reshapes them so the model sees tensors shaped `[L, B, D]` where `L` is sequence length (here often 1), `B` is batch size, and `D` is feature dim.

4) Model architecture (core)
- Spikformer (`Model/spikformer.py`)
  - A transformer-like block adapted for spiking-style processing.
  - It repeats an input vector over `T` time steps, applies spiking attention and spiking MLP blocks, then averages across time.
  - Produces a modulation signal (same shape as input) that the main model uses to modulate modality features.
- SentiCore (`Model/SentiCore_Model.py`)
  - Projects each modality into a shared `model_dim` space using linear layers.
  - Passes each projected modality through the `Spikformer` to get spiking modulation signals.
  - Applies spiking modulation (element-wise scaling) to each modality: x * (1 + s_softmax).
  - Optionally fuses modalities with `MultiAttnModel` (bidirectional multi-head cross attention stacks).
  - Concatenates fused modality vectors and applies an `fc` layer and a small `MLP` to produce final logits for emotion classes.
- MultiAttnModel (`Model/MultiAttn.py`)
  - A three-way fusion mechanism: separate stacks fuse text↔(audio,visual), audio↔(text,visual), visual↔(text,audio).
  - Contains gating to weight modalities before attention.
- MLP (`Model/MLP.py`)
  - Simple two-layer classifier that maps fused features to class logits.

5) Losses and training
- Train script: `Train/train_senticore.py` implements the training loop, scheduler, MLflow logging, and checkpointing.
- Losses:
  - SoftHGR (`Loss/SoftHGRLoss.py`): encourages high correlation between modality features while penalizing covariance (soft HGR objective).
  - MultiDSC (`Loss/MultiDSCLoss.py`): self-adjusting Dice loss to handle class imbalance.
  - Cross-Entropy: standard classification loss.
- Total loss is a weighted sum of (HGR, DSC, CE) — weights are set in the training `Config` and can be tuned.
- Optimizer: Adam with cosine annealing scheduler.

6) Finetuning and Hyperparameter Search
- Optuna integration: `finetuning/tune_optuna.py` runs Optuna to search learning rate, weight decay, grad clipping, and loss weights.
- MLflow: experiments and runs are logged to SQLite databases stored under `final/finetuning/`:
  - `snn_mlflow_finetune.db`: MLflow DB used during finetuning searches.
  - `snn_optuna_finetune.db`: storage used by Optuna to persist study trials.
  - `snn.db`: final training MLflow DB used during production training runs.

7) Inference / Production pipeline
- Location: `inference/` (kept separate from `final/` training code so it can be deployed).
- `inference/pipeline.py` builds the model architecture using values saved in `configs/inference_config.json`, loads a saved checkpoint, and exposes `InferencePipeline.predict(video_path)` which:
  1. Segments the video with Whisper (fast or faster-whisper implementation).
  2. Extracts text, audio, and visual features for each detected utterance using small encoder classes under `inference/preprocessing/` (RoBERTa, Wav2Vec2, ResNet50).
  3. Collates features and runs the model to get probabilities per emotion.
  4. Returns a list of `UtterancePrediction` objects with start/end/time, transcription, emotion, confidence, and full scores.
- Note: Inference intentionally requires a CUDA GPU. The pipeline raises an error if no GPU is available.

8) Checkpoints and config
- `checkpoints/` holds saved model states (created during training).
- `configs/inference_config.json` (created by `Train/save_config.py`) stores the final model architecture and hyperparameters needed to recreate the model for inference. `save_config.py` pulls parameters from an MLflow run and writes this file.

9) Usage notes and requirements (short)
- This codebase assumes a CUDA GPU is present for both training and inference.
- Feature extraction and model training require the following heavy libraries: PyTorch, transformers (RoBERTa, Wav2Vec2), torchvision, spikingjelly, faster-whisper or whisper, ffmpeg, av, facenet_pytorch (optional), mlflow, optuna, and others listed in `final/requirements.txt` or `final/requirements.txt` if present.
- MLflow SQLite DBs are stored under `final/finetuning/` by default. This keeps experiment data local to the project folder.

10) How to run (examples)
- Build features (phase 2) after you have segment lists in `preprocessing/phase1_segmentation`:
  - python preprocessing/phase2_features/phase2_pipeline.py
- Train (example):
  - python Train/train_senticore.py
- Save inference config from an MLflow run:
  - python Train/save_config.py --run-id <RUN_ID>
- Run inference on a video:
  - python inference/infer.py --video path/to/video.mp4 --weights path/to/checkpoint.pt --config configs/inference_config.json

If you want, I can also add a short quick-start script or fill in a requirements file next.

---
README created to explain the whole project in simple English. If you want any section expanded or rewritten even more simply, tell me which part.
