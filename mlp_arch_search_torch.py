import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from imblearn.over_sampling import SMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.base import BaseSampler
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

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

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(layer_sizes[-1], 1))  # Single output neuron
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_and_evaluate(model, train_loader, val_loader, epochs):
    def closure():
        optimizer.zero_grad()
        outputs = model(train_loader.dataset.tensors[0])
        loss = criterion(outputs.squeeze(), train_loader.dataset.tensors[1])
        loss.backward()
        return loss

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS(model.parameters())
    best_val_acc = 0
    metrics = {"tp": [],
               "tn": [],
               "fp": [],
               "fn": [],
               "uar": []}
    loss_values = []
    for epoch in range(epochs):
        # Training phase
        model.train()

        loss = optimizer.step(closure)
        loss_values.append(loss.cpu().detach().numpy())
        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x).squeeze()
                predictions = torch.sigmoid(outputs) > 0.5
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        conf_matrix = confusion_matrix(val_targets, val_predictions)
        tn, fp, fn, tp = conf_matrix.ravel()
        metrics["tp"].append(tp)
        metrics["tn"].append(tn)
        metrics["fp"].append(fp)
        metrics["fn"].append(fn)
        uar = tp / (tp + fn) + tn / (tn + fp)
        metrics["uar"].append(0.5 * uar)
    metrics["loss"] = loss_values
    return metrics

N_SEED = 42
datasets = Path("", "training_data")
# select computational device -> changed to CPU as it is faster for small datasets (as SVD)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
print(f"The {DEVICE} will be used for the computation..")
torch.set_default_dtype(torch.float64)


for dataset in datasets.iterdir():
    print(f"evaluating dataset {dataset}")
    # load dataset
    with open(dataset.joinpath("dataset_selected.pk"), "rb") as f:
        dataset_file = pickle.load(f)
    X = np.array(dataset_file["data"])
    y = np.array(dataset_file["labels"])
    # path where to store results
    results_path = Path(".", "results_mlp", dataset)
    # get the number of features
    input_size = X.shape[1]
    # define KAN architecture


    # Define the architecture variations
    mlp_archs = [
        [input_size, input_size * 2 - int(0.1 * input_size)],
        [input_size, input_size * 2 - int(0.2 * input_size)],
        [input_size, input_size * 2 - int(0.3 * input_size)],
        [input_size, input_size * 2 - int(0.4 * input_size)],
        [input_size, input_size * 2 - int(0.5 * input_size)],
        [input_size, input_size * 2 - int(0.6 * input_size)],
        [input_size, input_size * 2 - int(0.7 * input_size)],
        [input_size, input_size * 2 - int(0.8 * input_size)],
        [input_size, input_size * 2 - int(0.9 * input_size)],
        [input_size, input_size],
        [input_size, input_size - int(0.1 * input_size)],
        [input_size, input_size - int(0.2 * input_size)],
        [input_size, input_size - int(0.3 * input_size)],
        [input_size, input_size - int(0.4 * input_size)],
        [input_size, input_size - int(0.5 * input_size)],
        [input_size, input_size, input_size],
        [input_size, input_size, input_size - int(0.1 * input_size)],
        [input_size, input_size, input_size - int(0.2 * input_size)],
        [input_size, input_size, input_size - int(0.3 * input_size)],
        [input_size, input_size, input_size - int(0.4 * input_size)],
        [input_size, input_size, input_size - int(0.5 * input_size)],
        [input_size, input_size, input_size - int(0.6 * input_size)],
        [input_size, input_size, input_size - int(0.7 * input_size)],
        [input_size, input_size, input_size - int(0.8 * input_size)],
        [input_size, input_size, input_size - int(0.9 * input_size)]
    ]
    for arch in mlp_archs:
        best_uar = []
        torch.manual_seed(0)

        # create results directory for each dataset and evaluated architecture
        result_dir = results_path.joinpath(str(arch).replace(",", "_").replace(" ", ""))
        result_dir.mkdir(parents=True, exist_ok=True)
        # Monte Carlo cross-validation = split train/test 10 times
        print(f"evaluating {str(arch)}")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=32)
        idx = 0
        for train_index, test_index in skf.split(X, y):
            idx += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # KMeansSMOTE resampling. if fails 10x SMOTE resampling
            X_resampled, y_resampled = CustomSMOTE(kmeans_args={"random_state": N_SEED}).fit_resample(X_train, y_train)
            # MinMaxScaling
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_train_scaled = scaler.fit_transform(X_resampled)
            X_test_scaled = scaler.transform(X_test)

            # Create DataLoader for training and validation sets
            train_dataset = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_resampled).type(torch.float64))
            val_dataset = TensorDataset(torch.from_numpy(X_test_scaled), torch.from_numpy(y_test).type(torch.float64))
            train_loader = DataLoader(train_dataset, batch_size=len(y_train), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=len(y_test), shuffle=False)

            # create model
            model = MLP(arch)
            results = train_and_evaluate(model, train_loader, val_loader, 200)
            best_uar.append(np.max(results["uar"]))

            with open(result_dir.joinpath(f'mlp_res_{idx}.pickle'), "wb") as output_file:
                pickle.dump(results, output_file)
        print(f"mean uar: {np.mean(best_uar)}")