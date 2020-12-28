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

# Make some continuous features categorical
continuous.remove("MSSubClass")
categorical.append("MSSubClass")
# %%
# Concatenate list columns
listed = [
    "Condition1",
    "Condition2",
    "Exterior1st",
    "Exterior2nd",
    "BsmtFinType1",
    "BsmtFinType2",
]

for col in listed:
    if col in categorical:
        categorical.remove(col)
# %%
## Create pipeline
continuous_transformer = SimpleImputer(strategy="constant")

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse=False),
)

list_tranformer = CountVectorizer(analyzer=set)


def create_attributes(data):
    data["Attributes"] = [list(row[listed].astype(str)) for _, row in data.iterrows()]

    return data


preprocessor = make_pipeline(
    FunctionTransformer(create_attributes),
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
            ("list", list_tranformer, "Attributes"),
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
}

if not best_params:
    grid_search = GridSearchCV(
        make_pipeline(preprocessor, XGBRegressor()),
        {
            "xgbregressor__n_estimators": [100, 500, 1000],
            "xgbregressor__learning_rate": [0.01, 0.03],
            "xgbregressor__max_depth": [3, 6, 9],
        },
        scoring="neg_root_mean_squared_error",
        verbose=2,
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