"""
Script that performs grid search for SVM across various datasets.
Datasets are balanced via KMeansSMOTE and SMV is 10 fold cross-validated.
"""
import pickle
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from classifier_configs import get_classifier
from src.custom_metrics import unweighted_average_recall_score, bookmakers_informedness

N_SEED = 42
np.random.seed(N_SEED)

scoring_dict = {"mcc": make_scorer(matthews_corrcoef),
                "accuracy": make_scorer(accuracy_score),
                "recall": make_scorer(recall_score),
                "specificity": make_scorer(recall_score, pos_label=0),
                "gm": make_scorer(geometric_mean_score),
                "uar": make_scorer(unweighted_average_recall_score),
                "bm": make_scorer(bookmakers_informedness)}


def get_datasets_to_process(datasets_path: Path, results_data: Path):
    """
    Get datasets that were not evaluated.
    :param datasets_path: Path, path to training datasets
    :param results_data: Path, path to results folder (used to check which datasets were already evaluated)
    :param dataset_slice: int, slice of datasets to evaluate. If None all datasets will be evaluated.
    If tuple, slice will be used, e.g. (10, 100). If int, first n datasets will be evaluated.
    :return: list of datasets to evaluate
    """
    # get all datasets in training_data
    td = sorted([str(x.name) for x in datasets_path.iterdir()])

    # get all datasets, that were already evaluated
    tr = sorted([str(x.name) for x in results_data.iterdir()])
    # to perform gridsearch only for datasets that were not evaluated so far
    to_do = sorted(list(set(td) - set(tr)))

    return to_do


# pylint: disable=too-many-locals
def main(sex: str = "women",
         classifier="svm_poly"):
    """
    Main function for the classifier pipeline.

    :param sex: The sex for which the classifier is trained. Default is "women".
    :param classifier: The type of classifier to use. Default is "svm_poly".
    :param dataset_slice: The slice of the dataset to process. Default is None.
    """
    training_data = Path(".").joinpath("training_data", sex)
    results_data = Path(".").joinpath("results", classifier, sex)
    results_data.mkdir(exist_ok=True, parents=True)

    dataset = get_datasets_to_process(training_data, results_data)
    dataset = sorted(dataset)
    for training_dataset_str in tqdm.tqdm(dataset):
        results_file = results_data.joinpath(str(training_dataset_str))
        # path to training dataset
        training_dataset = training_data.joinpath(training_dataset_str)
        # print(f"evaluate {training_dataset}")
        results_data.joinpath(str(training_dataset.name)).mkdir(parents=True, exist_ok=True)
        # load dataset
        if not f"{training_dataset.name}" == "dataset_selected.pk":
            with open(training_data.joinpath(str(training_dataset.name), "dataset_selected.pk"), "rb") as f:
                train_set = pickle.load(f)
            dataset = {"X": np.array(train_set["data"]),
                       "y": np.array(train_set["labels"])}
            # imblearn pipeline perform the resampling only with the training dataset
            # and scaling according to training dataset
            pipeline, param_grid = get_classifier(classifier, random_seed=N_SEED)
            cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=N_SEED)
            # sklearn gridsearch with crossvalidation
            grid_search = GridSearchCV(pipeline, param_grid, cv=cross_validation, scoring=scoring_dict,
                                       n_jobs=-1, refit=False)
            grid_search.fit(dataset["X"], dataset["y"])
            # create folder with the training_dataset name to store results
            results_file.mkdir(exist_ok=True)
            # no need to write header again and again and again,...
            if results_file.joinpath("results.csv").exists():
                header = False
            else:
                header = True

            # dump gridsearch results to a results.csv
            print(pd.DataFrame(grid_search.cv_results_))

            pd.DataFrame(grid_search.cv_results_).round(6)[
                ["params", "mean_test_accuracy", "std_test_accuracy", "mean_test_recall", "std_test_recall",
                 "mean_test_specificity", "std_test_specificity", "mean_test_mcc", "std_test_mcc",
                 "mean_test_gm", "std_test_gm", "mean_test_uar", "std_test_uar", "mean_test_bm", "std_test_bm"]
                 ].to_csv(results_file.joinpath("results.csv"),
                index=False, mode="a",
                header=header, encoding="utf8", lineterminator="\n")

    # pylint: enable=too-many-locals


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(
        prog="classifier_pipeline.py",
        description="Perform grid search for different classifier across various datasets. "
    )
    parser.add_argument("classifier", type=str)
    parser.add_argument("sex", type=str)
    args = parser.parse_args()
    sex_to_compute = args.sex
    used_classifier = args.classifier
    main(sex_to_compute, used_classifier)
