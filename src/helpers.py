from sklearn import (
    svm,
    tree,
    linear_model,
    neighbors,
    naive_bayes,
    ensemble,
    discriminant_analysis,
    gaussian_process,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import warnings

# Configure some tools
plt.rcParams.update({"figure.figsize": (13, 10)})
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 50)


def get_score(model, X, y, cv_options={}):
    start_time = time.time()
    name = model.__class__.__name__
    name = model[1].__class__.__name__ if name == "Pipeline" else name
    print("Getting score for", name)

    scores = cross_val_score(model, X, y, **cv_options)
    mean = scores.mean()

    end_time = time.time()
    print("Scored", mean, "in", round(end_time - start_time), "s")

    return mean


def look_for_algorithm(
    X, y, preprocessor=None, ratio=1.0, enabled=False, cv_options={}
):
    if not enabled or not preprocessor:
        return

    cutoff = int(X.count().max() * ratio)

    algorithms = [
        # Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),
        # Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),
        # GLM
        linear_model.LogisticRegressionCV(),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.Perceptron(),
        linear_model.RidgeClassifierCV(),
        linear_model.SGDClassifier(),
        # Naives Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.CategoricalNB(),
        naive_bayes.ComplementNB(),
        naive_bayes.GaussianNB(),
        naive_bayes.MultinomialNB(),
        # Nearest Neighbor
        neighbors.KNeighborsClassifier(),
        # SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),
        # Trees
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),
        # Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
        # XGBoost
        XGBClassifier(),
    ]

    results = []
    for algorithm in algorithms:
        name = algorithm.__class__.__name__
        pipeline = make_pipeline(preprocessor, algorithm)

        results.append(
            {
                "algorithm": name,
                "score": get_score(pipeline, X[:cutoff], y[:cutoff], cv_options),
            }
        )

    sorted_results = pd.DataFrame(
        sorted(results, key=lambda result: result["score"], reverse=True)
    )

    sns.barplot(x="score", y="algorithm", data=sorted_results)

    plt.title("Machine Learning Algorithm Score \n")
    plt.xlabel("Cross Val Score (%)")
    plt.ylabel("Algorithm")


def draw_correlation_matrix(dataset):
    graph, ax = plt.subplots(figsize=(14, 12))

    return sns.heatmap(
        dataset.corr(),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        ax=ax,
        annot=True,
        annot_kws={"fontsize": 12},
    )


def split_predictor(dataset, predictor):
    dataset.dropna(axis=0, subset=[predictor], inplace=True)
    y = dataset[predictor]

    return [dataset.drop([predictor], axis=1, inplace=False), y]


def remove_near_constant(dataset, treshold=0.998, debug=False):
    near_constant = [
        column
        for column in dataset.columns
        if (dataset[column].value_counts(normalize=True).to_list()[0] or 0) > treshold
    ]

    if debug:
        print("Near constant ({}%):".format(treshold * 100), near_constant)

    return [dataset.drop(near_constant, axis=1, inplace=False), near_constant]


def remove_missing_data(dataset, treshold=0.95, debug=False):
    missing = [
        column
        for column in dataset.columns
        if (dataset[column].isnull().sum() / dataset[column].isnull().count())
        > treshold
    ]

    if debug:
        print("Missing data ({}%):".format(treshold * 100), missing)

    return [dataset.drop(missing, axis=1, inplace=False), missing]


def submit_predictions(submission):
    output = pd.DataFrame(submission)
    output.to_csv("submission.csv", index=False)


def get_features(
    X, arbitrary, debug=False, constant_treshold=0.998, missing_treshold=0.95
):
    [X, removed_near] = remove_near_constant(X, debug=debug, treshold=constant_treshold)
    [X, removed_missing] = remove_missing_data(
        X, debug=debug, treshold=missing_treshold
    )

    trashed = arbitrary + removed_near + removed_missing

    numerical_cols = (
        X.select_dtypes(include=["int64", "float64"])
        .columns.drop(trashed, errors="ignore")
        .tolist()
    )

    categorical_cols = (
        X.select_dtypes("object").columns.drop(trashed, errors="ignore").tolist()
    )

    return [numerical_cols, categorical_cols]


def search_parameters(pipeline, parameters, X, y):
    grid_search = GridSearchCV(pipeline, parameters)
    grid_search.fit(X, y)

    print(
        "Best score {:.4f} with parameters {}".format(
            grid_search.best_score_, grid_search.best_params_
        )
    )