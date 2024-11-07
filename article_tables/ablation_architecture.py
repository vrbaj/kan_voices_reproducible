"""This script utilzies results from main script to conudct an ablation study"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


SEXES = ["women", "men"]
BEST_ARCHS = ["126_89_2", "115_150_127"]
BEST_GRIDS = [5, 5]
BEST_DEGREES = [3, 5]

def calculate_mcc(tps, tns, fps, fns):
    """
    Calculate the Matthews correlation coefficient (MCC) for multiple sets of true positives, true negatives,
    false positives, and false negatives.
    :param tps: (list of int) List of true positive counts.
    :param tns: (list of int) List of true negative counts.
    :param fps: (list of int) List of false positive counts.
    :param tns: (list of int) List of false negative counts.
    :returns: float The maximum MCC value calculated from the provided lists.
    """
    return np.max([(tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) for
        tp, tn, fp, fn in zip(tps, tns, fps, fns)])

def main():
    """Ablation study - freezing grid and degree for KAN"""
    for sex, best_arch, best_grid, best_k in zip(SEXES, BEST_ARCHS, BEST_GRIDS, BEST_DEGREES):
        for param in ["grid", "degree"]:
            info = {"param": [], "mcc": []}
            if param == "grid":
                list_paths = list(Path(".").resolve().parents[0].glob(
                    f"results_mc_lamb0.001_g{best_grid}_k*/training_data/{sex}/*{best_arch}?"))
                other_param = "degree"
            else:
                list_paths = list(Path(".").resolve().parents[0].glob(
                    f"results_mc_lamb0.001_g*_k{best_k}/training_data/{sex}/*{best_arch}?"))
                other_param = "grid"
            # Iterate through experiments
            for path in list_paths:
                list_best_results = []
                for fold in path.glob("*.pickle"):
                    with open(fold, "rb") as f:
                        data = pickle.load(f)
                        list_best_results.append(
                            calculate_mcc(data["test_tp"], data["test_tn"], data["test_fp"], data["test_fn"]))

                info["mcc"].append(np.mean(list_best_results))
                if param == "grid":
                    info["param"].append(int(path.resolve().parents[2].name.replace("k", "").split("_")[-1]))
                else:
                    info["param"].append(int(path.resolve().parents[2].name.replace("g", "").split("_")[-2]))

            table = pd.DataFrame(info)
            table["Change of MCC"] = table["mcc"]
            print(table)
            print("")
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_title(f"Influence of {param.upper()} when {other_param.upper()} is fixed - {sex}")
            ax.scatter(table["param"], table["mcc"])
            ax.set_xlabel(f"{param.upper()} parameter")
            ax.set_ylabel("MCC")
            plt.tight_layout()
            fig.savefig(f"ablation_{sex}_{other_param}_fixed.png")

if __name__ == "__main__":
    main()
