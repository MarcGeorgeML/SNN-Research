import argparse
import optuna
import mlflow
import os
import sys
import json
from pathlib import Path

# Add parent folder (final/) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Train.train_senticore import Config, Trainer, set_seed

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class OptunaTrainer(Trainer):
    def setup_mlflow(self):
        mlflow.set_tracking_uri(f"sqlite:///snn_mlflow_finetune.db")
        mlflow.set_experiment("snn_finetune.db")

    def save_checkpoint(self, epoch, val_f1):
        # Disable checkpointing during Optuna to save disk
        return

def objective(trial, epochs):
    config = Config()
    
    # set features path
    config.feature_root = os.path.join(repo_root, "features")

    # Phase 1 search space
    config.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    config.grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0)

    # Loss weights (normalized)
    l_hgr = trial.suggest_float("loss_HGR_raw", 0.1, 0.6)
    l_dsc = trial.suggest_float("loss_DSC_raw", 0.1, 0.6)
    l_ce  = trial.suggest_float("loss_CE_raw", 0.1, 0.6)
    s = l_hgr + l_dsc + l_ce
    config.loss_HGR = l_hgr / s
    config.loss_DSC = l_dsc / s
    config.loss_CE  = l_ce  / s

    # Track normalized weights for visibility
    trial.set_user_attr("loss_HGR", config.loss_HGR)
    trial.set_user_attr("loss_DSC", config.loss_DSC)
    trial.set_user_attr("loss_CE",  config.loss_CE)

    # Optional: shorten epochs for faster search via env var
    config.epochs = epochs

    set_seed(config.seed)
    trainer = OptunaTrainer(config)
    trainer.train()

    return float(trainer.best_f1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    storage = "sqlite:///snn_optuna_finetune.db"
    study = optuna.create_study(
        study_name="senticore_phase1",
        direction="maximize",
        storage=storage,
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, args.epochs), n_trials=args.n_trials)
    # Save best config to JSON
    out_dir = Path(__file__).parent
    best = study.best_trial

    best_config = {
        "best_value": best.value,
        "params": best.params,
        "user_attrs": best.user_attrs,  # contains normalized loss weights
    }

    with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    # Optional: save all trials
    study.trials_dataframe().to_csv(out_dir / "all_trials.csv", index=False)

if __name__ == "__main__":
    main()
    
# python finetuning/tune_optuna.py --n-trials 100 --epochs 30
# mlflow ui --backend-store-uri sqlite:///snn_mlflow_finetune.db