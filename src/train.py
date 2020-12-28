# %% [code]
import pandas as pd
import seaborn as sns
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from helpers import *

# %% [code]
# Import the data
X_train = pd.read_csv("../input/home-data-for-ml-course/train.csv", index_col="Id")
X_test = pd.read_csv("../input/home-data-for-ml-course/test.csv", index_col="Id")

# %% [code]
## Split predictor
[X, y] = split_predictor(X_train, "SalePrice")

# %%
## Define correlated features
correlations = X_train.corr()

sns.heatmap(correlations, mask=correlations < 0.8, annot=True)

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

X[features].head()
# %% [code]
## Create pipeline
continuous_transformer = make_pipeline(SimpleImputer(strategy="constant"))

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = make_column_transformer(
    (
        continuous_transformer,
        continuous,
    ),
    (
        categorical_transformer,
        categorical,
    ),
)

estimator = XGBRegressor(learning_rate=0.1, random_state=0)
pipeline = make_pipeline(preprocessor, estimator)

# %%
from sklearn.model_selection import GridSearchCV

best_params = {
    "xgbregressor__learning_rate": 0.03,
    "xgbregressor__max_depth": 3,
    "xgbregressor__n_estimators": 1000,
}

if not best_params:
    grid_search = GridSearchCV(
        pipeline,
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
# %%
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