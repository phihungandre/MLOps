"""
predict.py

This module contains functions and classes related to prediction.
"""

import os
import json

import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("experiment_01")
mlflow.autolog()

if __name__ == "__main__":
    # find the latest and best model
    runs = mlflow.search_runs(experiment_ids=[1],order_by=["metrics.best_cv_score desc"])
    best_run = runs.head(1).to_dict(orient="records")[0]

    print(f"Best run id: ", best_run["run_id"])

    # save to environment variable
    os.environ["MLFLOW_RUN_ID"] = best_run["run_id"]

    # reload some data for predictions

    data = pd.read_csv("../data/dpe_processed_20250120.csv")
    data = data.sample(n=1, random_state=42).reset_index(drop=True)

    # save data to sample.json

    with open("../data/sample.json", "w") as f:
        json.dump(data.to_dict(orient="records"), f, indent=4)

    # load data as a PyfuncModel
    MODEL_URI = f"runs:/{best_run['run_id']}/best_estimator"
    loaded_model = mlflow.pyfunc.load_model(MODEL_URI)

    # make predictions

    loaded_model.predict(pd.DataFrame(data))
