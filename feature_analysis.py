import pickle
from pathlib import Path


import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# path to training datasets
datasets = Path("", "training_data", "women")
# select computational device -> changed to CPU as it is faster for small datasets (as SVD)
DEVICE = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(DEVICE)
print(f"The {DEVICE} will be used for the computation..")
to_evaluate = []
for dataset in list(datasets.iterdir()):
    if dataset.name in to_evaluate or len(to_evaluate) == 0:
        print(f"evaluating dataset {dataset}")
        # load dataset
        with open(dataset.joinpath("dataset.pk"), "rb") as f:
            dataset_file = pickle.load(f)
        X = np.array(dataset_file["data"])
        y = np.array(dataset_file["labels"])
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)
        print(f"X.shape {X.shape}")
        mut_info = mutual_info_classif(X, y, random_state=1)
        print(mut_info.shape)
        print(mutual_info_classif(X, y))
        to_drop = np.where(mut_info == 0.)[0]
        X = np.delete(X, to_drop, axis=1)
        print(f"After drop X {X.shape}")
        # selector = SelectKBest(mutual_info_classif, k=20)
        # X_reduced = selector.fit_transform(X, y)
        # X_reduced.shape
        # plt.plot(sorted(mut_info))
        # plt.show()
        dataset_to_dump = {"data": X, "labels": y}
        with open(dataset.parent.joinpath("dataset_selected.pk"), "wb") as f:
            pickle.dump(dataset_to_dump, f)

