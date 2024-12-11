"""
Script to analyze results of kan_arch_mc.py.
"""
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

np.seterr(all="raise")

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

    pickled_results_path = Path(".", "results_mlp")
    for dataset in sorted(pickled_results_path.iterdir()):
        sex = dataset.name
        print("Computing results for", sex)
        best_uar = 0
        for arch_result in tqdm(list(dataset.iterdir())):
            result_all_splits = {}
            for result in arch_result.glob("*.pickle"):
                # read result for a single train/test split
                # and save it into dictionary
                idx = int(result.stem.split("_")[-1])
                with open(result, "rb") as f:
                    result_all_splits[idx] = pickle.load(f)

            uars = np.array([result["uar"] for result in result_all_splits.values()])
            best_idx = np.argmax(np.mean(uars,axis=0))
            tps = np.array([result["tp"][best_idx] for result in result_all_splits.values()])
            fps = np.array([result["fp"][best_idx] for result in result_all_splits.values()])
            tns = np.array([result["tn"][best_idx] for result in result_all_splits.values()])
            fns = np.array([result["fn"][best_idx] for result in result_all_splits.values()])

            sensitivity = tps/(tps+fns)
            specificity = tns/(tns+fps)
            try:
                mcc = (tps * tns - fps * fns) / np.sqrt((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns))
            except FloatingPointError:
                # Zero division
                mcc = -1*np.ones_like(specificity)


            if np.mean(sensitivity/2+specificity/2) > best_uar:
                best_uar = np.mean(sensitivity/2+specificity/2)
                best_results[sex]["mcc"] = np.mean(mcc)
                best_results[sex]["mcc_std"] = np.std(mcc)
                best_results[sex]["sensitivity"] = np.mean(sensitivity)
                best_results[sex]["sensitivity_std"] = np.std(sensitivity)
                best_results[sex]["specificity"] = np.mean(specificity)
                best_results[sex]["specificity_std"] = np.std(specificity)
                best_results[sex]["bm"] = np.mean(sensitivity + specificity - 1)
                best_results[sex]["bm_std"] = np.std(sensitivity + specificity - 1)
                best_results[sex]["gm"] = np.mean(np.sqrt(sensitivity * specificity))
                best_results[sex]["gm_std"] = np.std(np.sqrt(sensitivity * specificity))
                best_results[sex]["uar"] = np.mean(sensitivity/2+specificity/2)
                best_results[sex]["uar_std"] = np.std(sensitivity/2+specificity/2)
                best_results[sex]["architecture"] = arch_result.name
    return best_results

if __name__ == "__main__":
    best_results_dict = main()
    print(f"BEST WOMEN: {best_results_dict["women"]["uar"]}")
    print(f"BEST MEN: {best_results_dict["men"]["uar"]}")
    print(f"BEST WOMEN DICT: {best_results_dict["women"]}")
    print(f"BEST MEN DICT: {best_results_dict["men"]}")
    print(best_results_dict)
