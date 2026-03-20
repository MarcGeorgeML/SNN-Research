import sqlite3
import pandas as pd

MLFLOW_DB = r"C:\Users\Marc\Desktop\Programming\SNN-Research\final\finetuning\snn_mlflow_finetune.db"

conn = sqlite3.connect(MLFLOW_DB)

# Pick experiment with most runs
exp_df = pd.read_sql_query("""
    SELECT e.experiment_id, e.name, COUNT(r.run_uuid) AS n_runs
    FROM experiments e
    LEFT JOIN runs r ON e.experiment_id = r.experiment_id
    GROUP BY e.experiment_id, e.name
    ORDER BY n_runs DESC
""", conn)

print(exp_df)

exp_id = exp_df.loc[0, "experiment_id"]
exp_name = exp_df.loc[0, "name"]
print("Using experiment:", exp_name)

exp_id = str(exp_df.loc[0, "experiment_id"])  # force text
runs_df = pd.read_sql_query(
    "SELECT run_uuid, name AS run_name, start_time, status FROM runs WHERE experiment_id = ?",
    conn, params=(exp_id,)
)

metrics_df = pd.read_sql_query(
    "SELECT run_uuid, key, value, step FROM metrics",
    conn
)

print("runs:", len(runs_df))
print("metrics total:", len(metrics_df))
print("metrics run_uuids sample:", metrics_df["run_uuid"].head(3).tolist())
print("runs run_uuids sample:", runs_df["run_uuid"].head(3).tolist())

df = metrics_df.merge(runs_df, on="run_uuid", how="inner")
print("joined rows:", len(df))

pivot = df.pivot_table(
    index=["run_uuid", "run_name", "step", "start_time", "status"],
    columns="key",
    values="value",
    aggfunc="last"
).reset_index()

conn.close()

pivot.to_csv("mlflow_metrics_history.csv", index=False)
print("Saved: mlflow_metrics_history.csv")