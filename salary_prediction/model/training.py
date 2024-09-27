import os
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, Ridge
from salary_prediction.data.processing import load_data, split_data
from salary_prediction.config.config import (
    PROCESSED_DATA,
    TRAINED_LINEAR_REGRESSOR,
    TRAINED_RIDGE_REGRESSOR,
    VALIDATION_TRAINING_DATA,
    EVALUATION_TEST_DATA,
)

df = load_data(PROCESSED_DATA).drop(columns="Unnamed: 0")

target = "Salary"
feature = df.columns[df.columns != target]

X = df[feature]
y = df[target]

X_train, X_test, y_train, y_test = split_data(X, y)


# Linear Regression Model
def train_linear_regressor(X_train, y_train):
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)

    return {"Model": lin_reg_model}


# Ridge (L2) Regression Model
def train_ridge_regressor(X_train, y_train):
    ridge_reg_model = Ridge(alpha=0.1, random_state=42)
    ridge_reg_model.fit(X_train, y_train)

    return {"Model": ridge_reg_model}


# Training the models in parallel to reduce the computing time
def parallel_model_training():
    tasks = [
        delayed(train_linear_regressor)(X_train, y_train),
        delayed(train_ridge_regressor)(X_train, y_train),
    ]

    trained_models = Parallel(n_jobs=-1, verbose=10)(tasks)
    return trained_models


trained_models_output = parallel_model_training()


# Save each trained model to its respective path
joblib.dump(trained_models_output[0]["Model"], TRAINED_LINEAR_REGRESSOR)
joblib.dump(trained_models_output[1]["Model"], TRAINED_RIDGE_REGRESSOR)

# Save validation data
joblib.dump(X_train, VALIDATION_TRAINING_DATA + "/X_train.pkl")
joblib.dump(y_train, VALIDATION_TRAINING_DATA + "/y_train.pkl")

# Save test data
joblib.dump(X_test, EVALUATION_TEST_DATA + "/X_test.pkl")
joblib.dump(y_test, EVALUATION_TEST_DATA + "/y_test.pkl")
