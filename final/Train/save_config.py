"""
save_config.py
--------------
Pulls hyperparameters from a finished MLflow run and saves them as an
inference_config.json ready for use with infer.py.

Usage
-----
    # Minimal — uses default tracking URI, prompts you to pick a run
    python save_config.py

    # Explicit run ID + tracking URI
    python save_config.py --run-id   8fa0157a2e664559a4d9694f0a3ef59d --tracking-uri sqlite:///path/to/snn.db --output   configs/inference_config.json
"""

import argparse
import json
import os
from pathlib import Path

from mlflow.tracking import MlflowClient


# ── default tracking URI — change this to match your setup ──────────────────
DEFAULT_TRACKING_URI = "sqlite:///C:/Users/Marc/Desktop/Programming/SNN-Research/final/finetuning/snn_mlflow_finetune.db"

# ── key remapping: MLflow param name → inference config key ─────────────────
# These are the keys logged by Config.to_dict() that need renaming for the
# model constructor.
KEY_REMAP = {
    "spike_tau"  : "tau",
    "spike_thr"  : "common_thr",
    "num_classes": "n_classes",
}

# ── keys to extract from MLflow params ──────────────────────────────────────
EXTRACT_KEYS = [
    "T",
    "spike_tau",     # → tau
    "spike_thr",     # → common_thr
    "model_dim",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "num_classes",   # → n_classes
]

# ── fixed architectural constants (hardcoded in build_model, never logged) ───
FIXED_CONSTANTS = {
    "depths"         : 2,
    "heads"          : 8,       # Spikformer heads
    "qk_scale"       : 0.125,
    "dataset"        : "custom",
    "multi_attn_flag": True,
    "roberta_dim"    : 768,
    "D_m_audio"      : 768,
    "D_m_visual"     : 2048,
    "dropout"        : 0,
}


# ── type coercion ─────────────────────────────────────────────────────────────
# MLflow stores all params as strings — we cast them back to the right types.
INT_KEYS   = {"T", "model_dim", "hidden_dim", "num_layers", "num_heads", "n_classes"}
FLOAT_KEYS = {"tau", "common_thr"}


def cast_value(key, value):
    if key in INT_KEYS:
        return int(value)
    if key in FLOAT_KEYS:
        return float(value)
    return value


# ─────────────────────────────────────────────────────────────────────────────

def list_runs(client, experiment_name="SentiCore_SNN_v2"):
    """Print all finished runs in the experiment."""

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"  No experiment named '{experiment_name}' found.")
        print("  Available experiments:")
        for exp in client.search_experiments():
            print(f"    ID={exp.experiment_id}  Name={exp.name}")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.best_val_f1 DESC"],
    )

    if not runs:
        print(f"  No runs found in experiment '{experiment_name}'.")
        return

    print(f"\n  {'RUN ID':<36}  {'NAME':<45}  {'best_val_f1':>11}  {'STATUS'}")
    print("  " + "-" * 105)

    for run in runs:
        f1    = run.data.metrics.get("best_val_f1", float("nan"))
        name  = run.info.run_name or ""
        rid   = run.info.run_id
        status= run.info.status
        print(f"  {rid:<36}  {name:<45}  {f1:>11.4f}  {status}")

    print()


def fetch_inference_config(client, run_id):
    """Pull params from MLflow, remap keys, cast types, merge fixed constants."""

    run = client.get_run(run_id)
    mlflow_params = run.data.params

    # check all required keys are present in the run
    missing = [k for k in EXTRACT_KEYS if k not in mlflow_params]
    if missing:
        raise KeyError(
            f"The following keys were not found in MLflow run {run_id}: {missing}\n"
            f"Available keys: {sorted(mlflow_params.keys())}"
        )

    inference_cfg = {}

    for mlflow_key in EXTRACT_KEYS:
        # rename if needed, otherwise keep the same key
        cfg_key = KEY_REMAP.get(mlflow_key, mlflow_key)
        raw_val = mlflow_params[mlflow_key]
        inference_cfg[cfg_key] = cast_value(cfg_key, raw_val)

    # merge fixed architectural constants (override nothing — these keys
    # are distinct from the extracted ones)
    inference_cfg.update(FIXED_CONSTANTS)

    return inference_cfg, run.info.run_name


def save_config(inference_cfg, output_path, run_id, run_name):
    """Write the config dict to a JSON file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # attach metadata so you always know which run this came from
    output = {
        "_meta": {
            "run_id"  : run_id,
            "run_name": run_name,
        },
        **inference_cfg,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Inference config saved → {output_path}")
    print(f"  Run : {run_name}")
    print(f"  ID  : {run_id}")
    print(f"\n  Config contents:")
    for k, v in inference_cfg.items():
        print(f"    {k:<20} : {v}")
    print()


def interactive_pick_run(client, experiment_name="SentiCore_SNN_v2"):
    """Print runs and ask user to paste a run ID."""

    list_runs(client, experiment_name)
    run_id = input("  Paste the Run ID you want to use: ").strip()

    if not run_id:
        raise ValueError("No run ID entered.")

    return run_id


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Save MLflow run hyperparameters as inference_config.json"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="8fa0157a2e664559a4d9694f0a3ef59d",
        help="MLflow run ID to export. If omitted, an interactive list is shown.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_TRACKING_URI})",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="SentiCore_SNN_v2",
        help="MLflow experiment name (default: SentiCore_SNN_v2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/inference_config.json",
        help="Output path for the JSON file (default: configs/inference_config.json)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all runs in the experiment and exit (no file saved)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    print(f"\n  Tracking URI : {args.tracking_uri}")
    print(f"  Experiment   : {args.experiment}")

    # ── just list runs and exit ───────────────────────────────────────────────
    if args.list:
        list_runs(client, args.experiment)
        return

    # ── resolve run ID ────────────────────────────────────────────────────────
    run_id = args.run_id
    if run_id is None:
        run_id = interactive_pick_run(client, args.experiment)

    # ── fetch + save ─────────────────────────────────────────────────────────
    print(f"\n  Fetching params for run: {run_id} …")
    inference_cfg, run_name = fetch_inference_config(client, run_id)
    save_config(inference_cfg, args.output, run_id, run_name)


if __name__ == "__main__":
    main()