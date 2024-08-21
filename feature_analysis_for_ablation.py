import pickle
from pathlib import Path


import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# path to training datasets
datasets = Path("", "training_data", "women")
to_iter = list(datasets.iterdir())
# select computational device -> changed to CPU as it is faster for small datasets (as SVD)
DEVICE = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(DEVICE)
print(f"The {DEVICE} will be used for the computation..")
to_evaluate = []
for to_elim in [1, -1]:
    if to_elim == 1:
        to_elim_text = "diff_pitch_elim"
    else:
        to_elim_text = "nan_elim"
    for dataset in to_iter:
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
            to_drop[to_elim] = True
            X = np.delete(X, to_drop, axis=1)
            print(f"After drop X {X.shape}")
            # selector = SelectKBest(mutual_info_classif, k=20)
            # X_reduced = selector.fit_transform(X, y)
            # X_reduced.shape
            # plt.plot(sorted(mut_info))
            # plt.show()
            dataset_to_dump = {"data": X, "labels": y}
            with open(dataset.parent.joinpath(f"dataset_selected_{to_elim_text}.pk"), "wb") as f:
                pickle.dump(dataset_to_dump, f)

