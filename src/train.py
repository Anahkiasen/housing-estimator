# %% [code]
import pandas as pd
import seaborn as sns
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBRegressor
from helpers import *

# %% [code]
# Import the data
X_train = pd.read_csv("../input/home-data-for-ml-course/train.csv", index_col="Id")
X_test = pd.read_csv("../input/home-data-for-ml-course/test.csv", index_col="Id")
# %%
# Drop outliers
outliers = X_train[(X_train.OverallQual > 9) & (X_train.SalePrice < 220000)].index
X_train.drop(outliers, inplace=True)
# %% [code]
## Split predictor
[X, y] = split_predictor(X_train, "SalePrice")
# %%
## Define correlated features
correlations = X_train.corr()

# sns.heatmap(correlations, mask=correlations < 0.8, annot=True)

# Sorted by correlation factor
# YearBuilt > GarageYrBuilt
# GarageArea > GarageCars
# GrLivArea > TotRmsAbvGrd
# TotalBsmtSF > 1stFlrSF
correlated = ["GarageYrBlt", "TotRmsAbvGrd", "GarageCars", "1stFlrSF"]

X_train.corr().SalePrice.sort_values()
# %% [code]
## Define data leaks
data_leaks = ["MoSold", "YrSold", "SaleType"]

## Define arbitrary features
arbitrary = []

## Select features
[continuous, categorical] = get_features(
    X,
    data_leaks + arbitrary + correlated,
    debug=True,
    constant_treshold=0.99,
    missing_treshold=0.94,
)

features = continuous + categorical
all_features = features + ["SalePrice"]
removed_features = X.columns.drop(features).tolist()
# %%
# Change the type of some columns
rating_columns = [
    "BsmtCond",
    "BsmtQual",
    "ExterCond",
    "ExterQual",
    "FireplaceQu",
    "GarageCond",
    "GarageQual",
    "HeatingQC",
    "KitchenQual",
]

finition_columns = ["BsmtFinType1", "BsmtFinType2"]

converted_ordinal = rating_columns + finition_columns + ["BsmtExposure", "Fence"]


def transform_ratings(data):
    for col in rating_columns:
        data[col] = data[col].map(
            {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
        )

    for col in finition_columns:
        data[col] = data[col].map(
            {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NA": 0}
        )

    data["BsmtExposure"] = data["BsmtExposure"].map(
        {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0}
    )

    data["Fence"] = data["Fence"].map(
        {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "NA": 0}
    )

    return data


continuous.remove("MSSubClass")
categorical.append("MSSubClass")

for converted in converted_ordinal:
    categorical.remove(converted)
    continuous.append(converted)
# %%
# Concatenate list columns
list_columns = [
    "Condition1",
    "Condition2",
    "Exterior1st",
    "Exterior2nd",
    # "BsmtFinType1",
    # "BsmtFinType2",
]


def create_attributes(data):
    data["Attributes"] = [
        list(row[list_columns].astype(str)) for _, row in data.iterrows()
    ]

    return data


for col in list_columns:
    if col in categorical:
        categorical.remove(col)
# %%
## Create pipeline
continuous_transformer = SimpleImputer(strategy="constant")

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse=False),
)


preprocessor = make_pipeline(
    make_pipeline(
        FunctionTransformer(create_attributes),
        FunctionTransformer(transform_ratings),
    ),
    ColumnTransformer(
        transformers=[
            (
                "continuous",
                continuous_transformer,
                continuous,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical,
            ),
            ("list", CountVectorizer(analyzer=set), "Attributes"),
        ]
    ),
)
# %%
# Test out pipeline
processed = preprocessor.fit_transform(X.copy())

applied_transformers = preprocessor[1].named_transformers_

processed_features = (
    continuous
    + (
        applied_transformers.categorical.named_steps.onehotencoder.get_feature_names(
            categorical
        ).tolist()
    )
    + applied_transformers.list.get_feature_names()
)

pd.DataFrame(processed, columns=processed_features)
# %%
# Find best hyperparameters
from sklearn.model_selection import GridSearchCV

best_params = {
    "xgbregressor__learning_rate": 0.03,
    "xgbregressor__max_depth": 3,
    "xgbregressor__n_estimators": 1000,
    "xgbregressor__min_child_weight": 2,
    "xgbregressor__gamma": 0,
    "xgbregressor__subsample": 0.75,
}

if not best_params:
    grid_search = GridSearchCV(
        make_pipeline(preprocessor, XGBRegressor()),
        {
            "xgbregressor__n_estimators": [100, 500, 1000],
            "xgbregressor__learning_rate": [0.01, 0.03],
            "xgbregressor__max_depth": [3, 6, 9],
            "xgbregressor__min_child_weight": [1, 2, 3],
            "xgbregressor__gamma": [0, 0.1],
            "xgbregressor__subsample": [0.25, 0.5, 0.75],
        },
        scoring="neg_root_mean_squared_error",
        verbose=10,
        n_jobs=-1,
    )

    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print(best_params)
# %%
# Score pipeline
pipeline = make_pipeline(preprocessor, XGBRegressor())
pipeline.set_params(**best_params)

get_score(pipeline, X, y, {"scoring": "neg_mean_absolute_error"})
# %%
# %% [code]
## Get predictions
pipeline.fit(X, y)
predictions = pipeline.predict(X_test)
submissions = {"Id": X_test.index, "SalePrice": predictions}

submit_predictions(submissions)