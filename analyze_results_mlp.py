"""
Script to analyze results of kan_arch_mc.py.
"""
import pickle
from pathlib import Path
import numpy as np


def main():
    """
    Main function to analyze results of MLP architecture search.
    :return: dict dictionary of best results, containing two dict for each sex
    """
    best_results = {
        "men": {
            "mcc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "gm": 0.0,
            "uar": 0.0,
            "bm": 0.0
        },
        "women": {
            "mcc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "gm": 0.0,
            "uar": 0.0,
            "bm": 0.0
        }
    }

    pickled_results_path = Path(".", "results_mlp", "training_data")
    for dataset in sorted(pickled_results_path.iterdir()):
        sex = dataset.name
        for arch_result in dataset.iterdir():
            best_metrics = {"mcc": [], "sensitivity": [], "specificity": [], "gm": [], "uar": [], "bm": []}
            for result in arch_result.glob("*.pickle"):
                # read result for a single train/test split
                with open(result, "rb") as f:
                    experiment_results = pickle.load(f)
                # compute UAR
                is_nan = False
                try:
                    mcc = [(tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                            for tp, tn, fp, fn
                            in zip(experiment_results["test_tp"],
                                   experiment_results["test_tn"],
                                   experiment_results["test_fp"],
                                   experiment_results["test_fn"])]
                    best_idx = np.argmax(mcc)
                    sensitivity = experiment_results["tp"][best_idx] / (experiment_results["tp"][best_idx] + experiment_results["fn"][best_idx])
                    specificity = experiment_results["tn"][best_idx] / (experiment_results["tn"][best_idx] + experiment_results["fp"][best_idx])

                    best_metrics["mcc"].append(np.max(mcc))
                    best_metrics["sensitivity"].append(sensitivity)
                    best_metrics["specificity"].append(specificity)
                    best_metrics["gm"].append(np.sqrt(sensitivity * specificity))
                    best_metrics["bm"].append(sensitivity + specificity - 1)
                    best_metrics["uar"].append((sensitivity + specificity) / 2)

                except KeyError:
                    print(experiment_results.keys())
                    print(f"error in {result.resolve()}")
                except ZeroDivisionError:
                    is_nan = True

            if not is_nan and np.mean(best_metrics["mcc"]) > best_results[sex]["mcc"]:
                    best_results[sex]["mcc"] = np.mean(best_metrics["mcc"])
                    best_results[sex]["mcc_std"] = np.std(best_metrics["mcc"])
                    best_results[sex]["sensitivity"] = np.mean(best_metrics["sensitivity"])
                    best_results[sex]["sensitivity_std"] = np.std(best_metrics["sensitivity"])
                    best_results[sex]["specificity"] = np.mean(best_metrics["specificity"])
                    best_results[sex]["specificity_std"] = np.std(best_metrics["specificity"])
                    best_results[sex]["bm"] = np.mean(best_metrics["bm"])
                    best_results[sex]["bm_std"] = np.std(best_metrics["bm"])
                    best_results[sex]["gm"] = np.mean(best_metrics["gm"])
                    best_results[sex]["gm_std"] = np.std(best_metrics["gm"])
                    best_results[sex]["uar"] = np.mean(best_metrics["uar"])
                    best_results[sex]["uar_std"] = np.std(best_metrics["uar"])
                    best_results[sex]["architecture"] = arch_result.name
                    best_results[sex]["settings"] = kan_settings.name
    return best_results

if __name__ == "__main__":
    best_results_dict = main()
    print(f"BEST WOMEN: {best_results_dict["women"]["uar"]}")
    print(f"BEST MEN: {best_results_dict["men"]["uar"]}")
    print(f"BEST WOMEN DICT: {best_results_dict["women"]}")
    print(f"BEST MEN DICT: {best_results_dict["men"]}")
