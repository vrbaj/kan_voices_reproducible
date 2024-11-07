"""
KAN arch search script.
"""
import os
import pickle
import random
from pathlib import Path

import torch
import numpy as np
from kan import KAN

from imblearn.over_sampling import SMOTE, KMeansSMOTE
from imblearn.base import BaseSampler
from sklearn.model_selection import  StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


N_SEED = 42

# Set the CUBLAS_WORKSPACE_CONFIG environment variable
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
def set_seed(seed):
    """
    Function to set seed for reproducibility.
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False




set_seed(N_SEED)  # You can choose any number you prefer

class CustomSMOTE(BaseSampler):
    """Class that implements KMeansSMOTE oversampling. Due to initialization of KMeans
    there are 10 tries to resample the dataset. Then standard SMOTE is applied.
    """
    _sampling_type = "over-sampling"

    def __init__(self, kmeans_args=None, smote_args=None):
        super().__init__()
        self.kmeans_args = kmeans_args if kmeans_args is not None else {}
        self.smote_args = smote_args if smote_args is not None else {}
        self.kmeans_smote = KMeansSMOTE(**self.kmeans_args)
        self.smote = SMOTE(**self.smote_args)

    # pylint:disable=broad-exception-caught,invalid-name
    def _fit_resample(self, X, y):
        resample_try = 0
        while resample_try < 10:
            try:
                X_res, y_res = self.kmeans_smote.fit_resample(X, y)
                return X_res, y_res
            except Exception:
                # dont care about exception, KmeansSMOTE failed
                self.kmeans_smote = KMeansSMOTE(random_state=resample_try)
                resample_try += 1
        X_res, y_res = self.smote.fit_resample(X, y)
        return X_res, y_res


def train_acc():
    """
    Train accuracy. That is how the PyKAN needs the metric functions.
    """
    return torch.mean((torch.argmax(model(dataset["train_input"]),
                                    dim=1) == dataset["train_label"]).float())


def test_acc():
    """
    Test accuracy. That is how the PyKAN needs the metric functions.
    """
    return torch.mean((torch.argmax(model(dataset["test_input"]),
                                    dim=1) == dataset["test_label"]).float())


def test_tp():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    return tp

def test_tn():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    return tn

def test_fp():
    """
    Specificity for the test. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN

    fp = ((predictions == 1) & (labels == 0)).sum().float()

    return fp

def test_fn():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    fn = ((predictions == 0) & (labels == 1)).sum().float()

    # Calculate recall
    return fn


def test_uar():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()

    # Calculate recall
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    uar = 0.5 * (recall + specificity)
    return uar


torch.manual_seed(32)

# since PyKAN 0.1.2 it is necessary to magically set torch default type to float64
# to avoid issues with matrix inversion during training with the LBFGS optimizer
torch.set_default_dtype(torch.float64)
# path to training datasets
datasets = Path("", "training_data")
# select computational device -> changed to CPU as it is faster for small datasets (as SVD)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
evaluated_ks = [3]
evaluated_grids = [9]
for k in evaluated_ks:
    for grid in evaluated_grids:
        for dataset in datasets.glob("men"):
            print(f"evaluating dataset {dataset}")
            # load dataset
            with open(dataset.joinpath("dataset_selected.pk"), "rb") as f:
                dataset_file = pickle.load(f)
            X = np.array(dataset_file["data"])
            y = np.array(dataset_file["labels"])
            # path where to store results
            results_path = Path(".", f"results_1layer_lamb0.001_g{grid}_k{k}_100epochs", dataset)
            # get the number of features
            input_size = X.shape[1]
            # define KAN architecture
            steps = list(np.linspace(0, 2, 11))
            kan_archs = []
            for first in steps:
                first_layer = input_size * 2 - int(first * input_size)
                if first_layer > 0:
                    kan_archs.append([input_size, first_layer, 2])
                    pass
                for second in steps:
                    second_layer = input_size * 2 - int(second * input_size)
                    if first_layer >= second_layer > 0:
                         kan_archs.append([input_size, first_layer, second_layer, 2])

            # iterate over KAN architectures and train for each dataset
            for arch in kan_archs:
                torch.manual_seed(0)

                # create results directory for each dataset and evaluated architecture
                result_dir = results_path.joinpath(str(arch).replace(
                    ",", "_").replace(" ", "").replace(
                    "[", "").replace("]", ""))
                result_dir.mkdir(parents=True, exist_ok=True)
                # Monte Carlo cross-validation = split train/test 10 times
                print(f"evaluating {str(arch)}")
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=N_SEED)
                idx = 0
                for train_index, test_index in skf.split(X, y):
                    idx += 1
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    # KMeansSMOTE resampling. if fails 10x SMOTE resampling
                    X_resampled, y_resampled = CustomSMOTE(
                        kmeans_args={"random_state": N_SEED}).fit_resample(X_train, y_train)
                    # MinMaxScaling
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    X_train_scaled = scaler.fit_transform(X_resampled)
                    X_test_scaled = scaler.transform(X_test)
                    print(np.isnan(np.min(X_train_scaled)), np.isnan(np.min(X_test_scaled)))
                    # KAN dataset format, load it to device
                    dataset = {"train_input": torch.from_numpy(X_train_scaled).to(DEVICE),
                               "train_label": torch.from_numpy(y_resampled).type(
                                   torch.LongTensor).to(DEVICE),
                               "test_input": torch.from_numpy(X_test_scaled).to(DEVICE),
                               "test_label": torch.from_numpy(y_test).type(
                                   torch.LongTensor).to(DEVICE)}

                    # create KAN model
                    model = KAN(width=arch, grid=grid, k=k, seed=N_SEED,
                                auto_save=False, save_act=True)
                    # load model to device
                    model.to(DEVICE)
                    # train model
                    results = model.fit(dataset, opt="LBFGS", lamb=0.001, steps=100, batch=-1,
                                        update_grid=True, metrics=(train_acc, test_acc, test_tn,
                                                                   test_tp, test_fn, test_fp,
                                                                   test_uar),
                                        loss_fn=torch.nn.CrossEntropyLoss())
                    # infotainment during training
                    print(f"final test acc: {results['test_acc'][-1]}"
                          f" mean test acc: {np.mean(results['test_acc'])}",
                          f"best test uar: {np.max(results["test_uar"])} ")

                    # dump results
                    with open(result_dir.joinpath(f'kan_res_{idx}.pickle'), "wb") as output_file:
                        pickle.dump(results, output_file)
