# %%
# Import dependencies
import pandas as pd
import seaborn as sns
from helpers import *
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# %%
# Import the data
data_train = pd.read_csv("../input/home-data-for-ml-course/train.csv", index_col="Id")
data_test = pd.read_csv("../input/home-data-for-ml-course/test.csv", index_col="Id")
data = data_train.append(data_test)

data
# %%
# Drop outliers
outliers = data_train[
    (data_train.OverallQual > 9) & (data_train.SalePrice < 220000)
].index
data_train.drop(outliers, inplace=True)
# %%
## Split predictor
[X, y] = split_predictor(data_train, "SalePrice")
# %%
## Examine correlated features
correlations = data_train.corr()

# Sorted by correlation factor
# YearBuilt > GarageYrBuilt
# GarageArea > GarageCars
# GrLivArea > TotRmsAbvGrd
# TotalBsmtSF > 1stFlrSF
correlated = ["GarageYrBlt", "TotRmsAbvGrd", "GarageCars", "1stFlrSF"]

# sns.heatmap(correlations, mask=correlations < 0.7, annot=True)
# correlations.SalePrice.sort_values()
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
data_leaks = ["MoSold", "YrSold", "SaleType", "SalePrice"]
arbitrary = []


def select_column_types(types, omit=[]):
    return (
        lambda data: data.select_dtypes(types)
        .columns.drop(data_leaks + arbitrary + correlated + omit, errors="ignore")
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
treshold = VarianceThreshold()

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
    treshold,
)
# %%
# Find best hyperparameters
best_params = {
    "xgbregressor__learning_rate": 0.03,
    "xgbregressor__max_depth": 3,
    "xgbregressor__n_estimators": 1000,
    "xgbregressor__min_child_weight": 2,
    "xgbregressor__gamma": 0,
    "xgbregressor__subsample": 0.75,
    "pipeline__variancethreshold__threshold": 0.001,
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
            "pipeline__variancethreshold__threshold": np.arange(
                start=0, stop=0.01, step=0.001
            ),
        },
        scoring="neg_root_mean_squared_error",
        verbose=10,
    )

    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print(best_params)
# %%
# Create pipeline
model = XGBRegressor()
pipeline = make_pipeline(preprocessor, model)
pipeline.set_params(**best_params)
# %%
# Score pipeline
get_score(pipeline, X, y, {"scoring": "neg_mean_absolute_error"})
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
    "Attribute_" + col for col in applied_transformers.list.get_feature_names()
]

processed_features = transformed_continuous + transformed_categorical + transformed_list
kept_features = [processed_features[i] for i in treshold.get_support(True)]
removed_features = [col for col in processed_features if col not in kept_features]
processed = pd.DataFrame(processed, columns=kept_features)

processed.head()

removed_features
# %%
# Score features
# test = X.copy()
# pipeline.fit(test, y)
# results = permutation_importance(pipeline, test, y, scoring="neg_mean_squared_error", n_jobs=-1)

# importances = pd.Series(results.importances_mean, index=test.columns).sort_values()
# negative = importances[importances < 0]
# important = importances[importances > 0]

# negative
# %%
## Get predictions
pipeline.fit(X, y)
predictions = pipeline.predict(data_test)
submissions = {"Id": data_test.index, "SalePrice": predictions}

submit_predictions(submissions)
