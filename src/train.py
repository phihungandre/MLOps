"""
train.py

This module contains the training of the random forest model.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import mlflow

REMOTE_SERVER_URI = "http://localhost:8088"
mlflow.set_tracking_uri(REMOTE_SERVER_URI)

mlflow.set_experiment("experiment_01")

mlflow.sklearn.autolog()

INPUT_FILE = "../data/dpe_processed_20250120.csv"
data = pd.read_csv(INPUT_FILE)

data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
assert y.name == "etiquette_dpe"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.drop(columns=["n_dpe"])
id_test = list(X_test["n_dpe"])
X_test = X_test.drop(columns=["n_dpe"])

rf = RandomForestClassifier()

# Define the parameter grid
param_grid = {
    "n_estimators": [200, 300],  # Number of trees
    "max_depth": [10],  # Maximum depth of the trees
    "min_samples_leaf": [1, 5],  # Maximum depth of the trees
}

# Setup GridSearchCV with k-fold cross-validation
cv = KFold(n_splits=3, random_state=84, shuffle=True)

grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=cv, scoring="accuracy", verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")
print(f"Best model: {grid_search.best_estimator_}")

# Evaluate on the test set
yhat = grid_search.predict(X_test)
print(classification_report(y_test, yhat))

probabilities = grid_search.predict_proba(X_test)

predictions = pd.DataFrame()
predictions.index = id_test
predictions["prob"] = np.max(probabilities, axis=1)
predictions["yhat"] = yhat
predictions["y"] = y_test.values
print(predictions.head())

# feature importance
feature_importances = grid_search.best_estimator_.feature_importances_
feature_names = X_train.columns

# Create a dictionary mapping feature names to their importance
importance_dict = dict(zip(feature_names, feature_importances))
importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

print(importance_dict)
