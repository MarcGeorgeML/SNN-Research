import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# Point to the MLflow DB that contains the run (final/finetuning)
finetuning_dir = Path(__file__).resolve().parent.parent / "finetuning"
db_path = finetuning_dir / "snn_mlflow_finetune.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

run_id = "8fa0157a2e664559a4d9694f0a3ef59d"
expected_name = "SentiCore_T8_dim256_h4_lr4.9334886565290195e-05_bs32_20260320_132547"

client = MlflowClient()
run = client.get_run(run_id)

print("Run name:", run.info.run_name)
print("Matches expected:", run.info.run_name == expected_name)
print("\nParams:")
for k, v in run.data.params.items():
    print(f"  {k}: {v}")