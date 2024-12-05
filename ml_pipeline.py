"""
Script that performs grid search for SVM across various datasets.
Datasets are balanced via KMeansSMOTE and SMV is 10 fold cross-validated.
"""
import pickle
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from ml_classifier_configs import get_classifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

scoring_dict = {"mcc": matthews_corrcoef,
                "accuracy": accuracy_score,
                "recall": recall_score,
                "specificity": (lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=0)),
                "gm": geometric_mean_score,
                "uar":  (lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred, adjusted=False)),
                "bm":  (lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred, adjusted=True))}


# pylint: disable=too-many-locals
def main(sex: str = "women",
         classifier="svm_poly"):
    """
    Main function for the classifier pipeline.

    :param sex: The sex for which the classifier is trained. Default is "women".
    :param classifier: The type of classifier to use. Default is "svm_poly".
    """
    training_data = Path(".").joinpath("training_data", sex)
    results_data = Path(".").joinpath("results", sex, classifier)
    results_data.mkdir(exist_ok=True, parents=True)

    pipeline, param_grid = get_classifier(classifier, random_seed=RANDOM_SEED)

    data = np.load(training_data.joinpath("datasets_selected.npz"))
    X_train=data['X_train']
    y_train=data['y_train']
    X_test=data['X_test']
    y_test=data['y_test']
    X_val=data['X_val']
    y_val=data['y_val']


    for params in tqdm.tqdm(list(ParameterGrid(param_grid))):
        pipeline.set_params(**params)
        pipeline.fit(X_train,y_train)
        y_pred = pipeline.predict(X_val)
        results = {"params": json.dumps(params)}
        for name,scorer in scoring_dict.items():
            results[name+"_val"] = scorer(y_val, y_pred)
        if 'result_table' not in locals():
            result_table = pd.DataFrame.from_dict([results])
            continue
        result_table = pd.concat([result_table, pd.DataFrame.from_dict([results])], ignore_index=True)

    result_table.round(6).to_csv(results_data.joinpath("results.csv"),
        index=False, header=True, encoding="utf8", lineterminator="\n")

    # find the row with the highest mcc
    best_row = result_table.iloc[result_table["mcc_val"].idxmax()]
    print(f'Best classificator has MCC={best_row["mcc_val"]} and UAR={best_row["uar_val"]}')
    best_params = json.loads(best_row["params"])
    print(best_params)
    pipeline.set_params(**best_params)
    pipeline.fit(X=np.concatenate((X_train,X_val), axis=0),
                 y=np.concatenate((y_train,y_val), axis=0))
    y_pred_test = pipeline.predict(X_test)
    results_final = {
        "classifier": classifier
    }
    results_final.update(best_row.to_dict())

    for name, scorer in scoring_dict.items():
        results_final[name+"_test"] = scorer(y_test, y_pred_test)

    X = np.concatenate((X_train, X_val, X_test))
    y = np.concatenate((y_train, y_val, y_test))

    scorers = {name: make_scorer(scorer) for name, scorer in scoring_dict.items()}

    cv_results = cross_validate(pipeline, X, y,
                                scoring=scorers,
                                cv=10, n_jobs=-1)
    print(pipeline.get_params())
    for name, values in cv_results.items():
        if not 'test_' in name:
            continue
        score_name = name.split("_")[-1]
        results_final[score_name+"_cv_mean"] = np.mean(values)
        results_final[score_name+"_cv_std"] = np.std(values)

    final_results_file = results_data.parent.joinpath("results_final.csv")

    pd.DataFrame.from_dict([results_final]).round(6).to_csv(
        final_results_file,
        index=False, mode="a",
        header= not final_results_file.exists(),
        encoding="utf8", lineterminator="\n")


if __name__ == "__main__":
    for classifier in ["svm_poly","svm_rbf","knn","gauss_nb","random_forest","adaboost"]:
        for sex in ["women", "men"]:
            print(f"Computing {classifier} for {sex}")
            main(sex, classifier)
