import os

import mlflow
from dotenv import load_dotenv

load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if not tracking_uri:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set in .env")

mlflow.set_tracking_uri(tracking_uri)

print("Tracking URI:")
print(mlflow.get_tracking_uri())

print("\nStarting MLflow run...")

with mlflow.start_run(run_name="azure-connection-test"):
    mlflow.log_param("test", "Azure ML connection")
    mlflow.log_param("environment", "local")

    mlflow.log_metric("rmse", 19.05)
    mlflow.log_metric("mae", 9.05)
    mlflow.log_metric("r2", 0.9649)

print("\nSUCCESS: Azure ML MLflow connection is working.")