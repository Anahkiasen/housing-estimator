from learntools.core import binder
from learntools.ml_intermediate.ex4 import *
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import optparse
import os
import pandas as pd
import time

if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")

binder.bind(globals())
print("Setup Complete")

# 1. Prepare the data
########################################

# Read the data
X_train_full = pd.read_csv("../input/train.csv", index_col="Id")
X_test_full = pd.read_csv("../input/test.csv", index_col="Id")

# Remove rows with missing target
X_train_full.dropna(axis=0, subset=["SalePrice"], inplace=True)

# Separate target from predictors
y = X_train_full.SalePrice
X_train_full.drop(["SalePrice"], axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
cardinal_categorical_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"
]

# Select numerical columns
numerical_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].dtype in ["int64", "float64"]
]

# Keep selected columns only
selected_cols = cardinal_categorical_cols + numerical_cols
X = X_train_full[selected_cols].copy()
X_test = X_test_full[selected_cols].copy()


# 2. Create prediction pipeline
########################################

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy="constant")

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_transformer, numerical_cols),
        ("categorical", categorical_transformer, cardinal_categorical_cols),
    ]
)


# 3. Cross-validation
########################################

model_configuration = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 0}
validation_range = np.arange(0.05, 0.3, 0.05)
validation_argument = "learning_rate"
validate = False
submit = True


def get_score(validation_configuration):
    start_time = time.time()
    print("Getting score for", validation_configuration)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(**model_configuration, **validation_configuration),
            ),
        ]
    )

    scores = -1 * cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error",
        fit_params={
            # "model__early_stopping_rounds": 5,
        },
    )

    mean = scores.mean()

    end_time = time.time()
    print("Scored", mean, "in", round(end_time - start_time), "s")

    return mean


# Evaluate the model
if validate:
    results = {
        value: get_score({validation_argument: value}) for value in validation_range
    }

    plt.plot(list(results.keys()), list(results.values()))
    plt.show()
elif not submit:
    get_score({})


# 4. Testing
########################################


# Bundle preprocessing and modeling code in a pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBRegressor(**model_configuration),
        ),
    ]
)

# Preprocessing of training data, fit model
pipeline.fit(X, y)

# Preprocessing of validation data, get predictions
predictions = pipeline.predict(X_test)

## Save test predictions to file
output = pd.DataFrame({"Id": X_test.index, "SalePrice": predictions})
output.to_csv("submission.csv", index=False)