# %% [code]
import pandas as pd
import seaborn as sns
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBRegressor
from helpers import *

plt.rcParams.update({"figure.figsize": (13, 10)})
# %% [code]
# Import the data
X_train = pd.read_csv("../input/home-data-for-ml-course/train.csv", index_col="Id")
X_test = pd.read_csv("../input/home-data-for-ml-course/test.csv", index_col="Id")

# %% [code]
## Split predictor
[X, y] = split_predictor(X_train, "SalePrice")

# %%
# Draw correlations
correlations = X_train.corr()

sns.heatmap(correlations, mask=correlations < 0.8)
# %% [code]
## Define data leaks
data_leaks = ["MoSold", "YrSold", "SaleType"]

## Define arbitrary features
arbitrary = []

## Define correlated features
correlated = ["GarageYrBlt", "TotRmsAbvGrd", "GarageCars", "1stFlrSF"]

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

estimator = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0)

pipeline = make_pipeline(preprocessor, estimator)

score = get_score(pipeline, X, y, {"scoring": "neg_mean_absolute_error", "cv": 5})

"{:,.0f}".format(score * -1)
# %%
# %% [code]
## Search for algorithm
look_for_algorithm(
    X,
    y,
    enabled=False,
    ratio=0.5,
    preprocessor=preprocessor,
    cv_options={"cv": 2, "scoring": "neg_mean_absolute_error"},
)


# %% [code]
## Train algorithm
pipeline.fit(X, y)


# %% [code]
## Get predictions
predictions = pipeline.predict(X_test)
submissions = {"Id": X_test.index, "SalePrice": predictions}

submit_predictions(submissions)

# %%