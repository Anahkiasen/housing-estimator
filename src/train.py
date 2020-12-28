# %%
# Import dependencies
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBRegressor
from helpers import *

# %%
# Import the data
X_train = pd.read_csv("../input/home-data-for-ml-course/train.csv", index_col="Id")
X_test = pd.read_csv("../input/home-data-for-ml-course/test.csv", index_col="Id")
X_all = X_train.append(X_test)

X_all
# %%
# Drop outliers
outliers = X_train[(X_train.OverallQual > 9) & (X_train.SalePrice < 220000)].index
X_train.drop(outliers, inplace=True)
# %%
## Split predictor
[X, y] = split_predictor(X_train, "SalePrice")
# %%
## Examine correlated features
correlations = X_train.corr()

# sns.heatmap(correlations, mask=correlations < 0.7, annot=True)

# Sorted by correlation factor
# YearBuilt > GarageYrBuilt
# GarageArea > GarageCars
# GrLivArea > TotRmsAbvGrd
# TotalBsmtSF > 1stFlrSF
correlated = ["GarageYrBlt", "TotRmsAbvGrd", "GarageCars", "1stFlrSF"]

X_train.corr().SalePrice.sort_values()
# %%
# Convert some features to their proper types
def transform_ordinal(data):
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

    data["MSSubClass"] = data["MSSubClass"].astype(str)

    return data


# %%
# Create new features
def create_features(data):
    list_columns = [
        "Condition1",
        "Condition2",
        "Exterior1st",
        "Exterior2nd",
    ]

    data["Attributes"] = [
        list(row[list_columns].astype(str)) for _, row in data.iterrows()
    ]

    data["TotalSF"] = data.TotalBsmtSF + data["1stFlrSF"] + data["2ndFlrSF"]

    data["YearBuilts"] = data.YearBuilt + data.YearRemodAdd

    data["TotalBathrooms"] = (
        data.FullBath
        + (0.5 * data.HalfBath)
        + data.BsmtFullBath
        + (0.5 * data.BsmtHalfBath)
    )

    data["TotalPorchSF"] = (
        data.OpenPorchSF
        + data["3SsnPorch"]
        + data.EnclosedPorch
        + data.ScreenPorch
        + data.WoodDeckSF
    )

    data["TotalQual"] = (
        data.OverallQual
        + data.ExterQual
        + data.BsmtQual
        + data.KitchenQual
        + data.GarageQual
    )

    return data.drop(list_columns, axis="columns")


# %%
# Backfill missing features
def backfill_missing(data):
    no_garages = data[data.GarageArea == 0].index
    no_basement = data[data.TotalBsmtSF == 0].index
    no_fireplace = data[data.Fireplaces == 0].index
    no_pool = data[data.PoolArea == 0].index

    garage_categorical = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
    garage_ordinal = ["GarageCars", "GarageArea", "GarageYrBlt"]
    basement_categorical = [
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
    ]
    basement_ordinal = [
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
    ]

    data.loc[no_garages, garage_categorical] = "NA"
    data.loc[no_pool, ["PoolQC"]] = "NA"
    data.loc[no_fireplace, ["FireplaceQu"]] = "NA"
    data.loc[
        no_basement,
        basement_categorical,
    ] = "NA"

    data.loc[no_garages, garage_ordinal] = 0
    data.loc[no_basement, basement_ordinal] = 0

    data["MSZoning"] = data.groupby("MSSubClass")["MSZoning"].transform(
        lambda x: x.fillna(x.mode()[0])
    )

    return data


# %%
# Drop near-constant and missing features
def remove_unused_features(data):
    ## Define data leaks
    data_leaks = ["MoSold", "YrSold", "SaleType", "SalePrice"]

    ## Define arbitrary features
    arbitrary = []

    ## Select features
    [continuous, categorical] = get_features(
        data,
        data_leaks + arbitrary + correlated,
        debug=False,
        constant_treshold=0.99,
        missing_treshold=0.94,
    )

    return data[continuous + categorical]


def select_column_types(types, omit=[]):
    return (
        lambda data: remove_unused_features(data)
        .select_dtypes(types)
        .columns.drop(omit)
        .tolist()
    )


# %%
# Assemble transformers
data_transformer = make_pipeline(
    FunctionTransformer(backfill_missing),
    FunctionTransformer(transform_ordinal),
    FunctionTransformer(create_features),
)

transformed = data_transformer.fit_transform(X.copy())
# %%
## Assemble preprocessors
from sklearn.preprocessing import StandardScaler

continuous_transformer = make_pipeline(
    SimpleImputer(strategy="constant"), StandardScaler()
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse=False),
)

list_features = ["Attributes"]
continuous_features = select_column_types(["int", "float"])
categorical_features = select_column_types(["object"], list_features)

preprocessor = make_pipeline(
    data_transformer,
    ColumnTransformer(
        transformers=[
            (
                "continuous",
                continuous_transformer,
                continuous_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
            ("list", CountVectorizer(analyzer=set), list_features[0]),
        ]
    ),
)
# %%
# Preview processed dataset
processed = preprocessor.fit_transform(X.copy(), y)

applied_transformers = preprocessor.named_steps.columntransformer.named_transformers_

transformed_continuous = continuous_features(transformed)

transformed_categorical = (
    applied_transformers.categorical.named_steps.onehotencoder.get_feature_names(
        categorical_features(transformed)
    ).tolist()
)

transformed_list = [
    "Attribute" + col for col in applied_transformers.list.get_feature_names()
]

processed_features = transformed_continuous + transformed_categorical + transformed_list
processed = pd.DataFrame(processed, columns=processed_features)
processed.head()

processed["SalePrice"] = y
# %% Create pipeline
pipeline = make_pipeline(preprocessor, XGBRegressor())
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
        pipeline,
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
pipeline.set_params(**best_params)

get_score(pipeline, X, y, {"scoring": "neg_mean_absolute_error"})
# %%
# %%
## Get predictions
pipeline.fit(X, y)
predictions = pipeline.predict(X_test)
submissions = {"Id": X_test.index, "SalePrice": predictions}

submit_predictions(submissions)
