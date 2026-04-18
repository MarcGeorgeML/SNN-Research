import mlflow
from mlflow.tracking import MlflowClient

# Point to the MLflow DB that contains the run
mlflow.set_tracking_uri(
    "sqlite:///C:/Users/Marc/Desktop/Programming/SNN-Research/final/finetuning/snn_mlflow_finetune.db"
)

run_id = "8fa0157a2e664559a4d9694f0a3ef59d"
expected_name = "SentiCore_T8_dim256_h4_lr4.9334886565290195e-05_bs32_20260320_132547"

client = MlflowClient()
run = client.get_run(run_id)

print("Run name:", run.info.run_name)
print("Matches expected:", run.info.run_name == expected_name)
print("\nParams:")
for k, v in run.data.params.items():
    print(f"  {k}: {v}")