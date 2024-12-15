import pickle
from pathlib import Path
import numpy as np

results_path = Path(".", "results_kan")

uar_results = {"women": [],
               "men": []}
uar_train_results = {"women": [],
                     "men": []}
for hyper_settings in results_path.iterdir():
    for sex in hyper_settings.iterdir():

        for arch in sex.iterdir():
            results = []
            train_results = []
            for fold in arch.iterdir():
                with open(fold, "rb") as f:
                    data = pickle.load(f)
                    results.append(data["test_uar"])
                    train_results.append(data["train_uar"])
            uar_results[sex.stem].append(np.mean(results))
            uar_train_results[sex.stem].append(np.mean(train_results))
print(f"Women best results: {np.max(uar_results['women'])}")
print(f"Women worst results: {np.min(uar_results['women'])}")

with open("one_epoch_results.pkl", "wb") as f:
    pickle.dump([uar_results, uar_train_results], f)
