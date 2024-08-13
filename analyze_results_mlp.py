"""
Script to analyze results of kan_arch_mc.py.
"""
import pickle
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    # path to results to analyze. It is expected that directory contains multiple
    # folders with different dataset and each folder contains subfolder
    # representing various architectures (kan_arch_mc.py).
    historical_best_men = []
    historical_best_women = []
    pickled_results_path = Path(".", "results_mlp", "training_data")
    for dataset in sorted(pickled_results_path.iterdir()):
        for arch_result in dataset.iterdir():
            print(f"evaluating architecture {arch_result}")
            best_uar = []
            for result in arch_result.glob("*.pickle"):
                # read result for a single train/test split
                with open(result, "rb") as f:
                    experiment_results = pickle.load(f)
                # compute UAR
                try:
                    # uar = [(recall + specificity) / 2 for recall, specificity in
                    #        zip(experiment_results["test_recall"],
                    #            experiment_results["test_specificity"])]
                    # select best UAR for this train/test split
                    best_uar.append(np.max(experiment_results["uar"]))

                except KeyError:
                    print("fuckedup")
            # prepare results to print
            TO_PRINT = " ".join([f"{num:.4f}" for num in best_uar])
            # print the BEST UAR for each split and MEAN of BEST UAR
            print(f"The best UAR for each split: {TO_PRINT}")
            print(f"Mean uar: {np.mean(best_uar)}")
            if dataset.name == "men":
                historical_best_men.append(np.mean(best_uar))
            else:
                historical_best_women.append(np.mean(best_uar))
    print(f"BEST WOMEN: {np.max(historical_best_women)}")
    print(f"BEST MEN: {np.max(historical_best_men)}")