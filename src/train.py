import os
import pandas as pd
import numpy as np
from learntools.core import binder
from learntools.ml_intermediate.ex1 import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")

binder.bind(globals())

print("Setup Complete")

# Read the data
samples_full = pd.read_csv("../input/train.csv", index_col="Id")
samples_submit_full = pd.read_csv("../input/test.csv", index_col="Id")


# Remove rows with missing target, separate target from predictors
samples_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
labels = samples_full.SalePrice
samples_full.drop(["SalePrice"], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
samples = samples_full.select_dtypes(exclude=["object"])
samples_submit = samples_submit_full.select_dtypes(exclude=["object"])

imputer = SimpleImputer(strategy="median")
imputed_samples = pd.DataFrame(imputer.fit_transform(samples))
imputed_samples_submit = pd.DataFrame(imputer.transform(samples_submit))

# Break off validation set from training data
(
    imputed_samples_train,
    imputed_samples_check,
    labels_train,
    labels_check,
) = train_test_split(imputed_samples, labels, train_size=0.8, random_state=0)


def score_model(model):
    model.fit(imputed_samples_train, labels_train)
    predictions = model.predict(imputed_samples_check)

    return round(mean_absolute_error(predictions, labels_check), 2)


# results = {}
# for value in np.arange(2, 10, 1):
##for value in ["mae", "mse"]:
#    print('Evaluating ', value)
#    model = RandomForestRegressor(
#        n_estimators=71,min_samples_split=5, criterion="mae", max_depth=18, random_state=0
#    )

#    results[value] = score_model(model)

# results_by_score = sorted(results.items(), key=lambda x: x[1])
# print(results_by_score)

model = RandomForestRegressor(
    n_estimators=71, min_samples_split=5, criterion="mae", max_depth=18, random_state=0
)

model.fit(imputed_samples, labels)
predictions = model.predict(imputed_samples_submit)

# Save predictions in format used for competition scoring
output = pd.DataFrame({"Id": samples_submit.index, "SalePrice": predictions})
output.to_csv("submission.csv", index=False)