import pickle
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from imblearn.base import BaseSampler
from torch.utils.data import DataLoader, TensorDataset

from src.customsmote import CustomSMOTE


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


N_SEED = 42 # You can choose any number you prefer

# select computational device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The {DEVICE} will be used for the computation..")
torch.set_default_dtype(torch.float64)

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


def train_and_evaluate(model, train_loader, val_loader, epochs, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS(model.parameters())
    metrics = {"tp": [],
               "tn": [],
               "fp": [],
               "fn": [],
               "uar": []}
    loss_values = []
    for _ in range(epochs):
        # Training phase
        model.train()

        def closure():
            optimizer.zero_grad()
            outputs = model(train_loader.dataset.tensors[0].to(device))
            loss = criterion(outputs.squeeze(), train_loader.dataset.tensors[1].to(device))
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_values.append(loss.cpu().detach().numpy())

        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x.to(device)).squeeze()
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


def main():
    datasets = Path("", "training_data")
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
        # define MLP architecture
        steps = list(np.linspace(0, 2, 21))
        mlp_archs = []
        for first in steps:
            first_layer = input_size * 2 - int(first * input_size)
            if first_layer > 0:
                mlp_archs.append([input_size, first_layer, 2])
            for second in steps:
                second_layer = input_size * 2 - int(second * input_size)
                if first_layer >= second_layer > 0:
                    mlp_archs.append([input_size, first_layer, second_layer, 2])
        print(mlp_archs)

        for arch in mlp_archs:
            best_uar = []
            set_seed(N_SEED)

            # create results directory for each dataset and evaluated architecture
            result_dir = results_path.joinpath(str(arch).replace(",", "_").replace(" ", ""))
            result_dir.mkdir(parents=True, exist_ok=True)
            print(f"evaluating {str(arch)}")
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=N_SEED)
            idx = 0
            for train_index, test_index in skf.split(X, y):
                idx += 1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # KMeansSMOTE resampling. if fails 10x SMOTE resampling
                X_resampled, y_resampled = CustomSMOTE(random_state=N_SEED).fit_resample(X_train, y_train)
                # MinMaxScaling
                scaler = MinMaxScaler(feature_range=(-1, 1))
                X_train_scaled = scaler.fit_transform(X_resampled)
                X_test_scaled = scaler.transform(X_test)

                # Create DataLoader for training and validation sets
                train_dataset = TensorDataset(torch.from_numpy(X_train_scaled).to(DEVICE),
                                            torch.from_numpy(y_resampled).type(torch.float64).to(DEVICE))
                val_dataset = TensorDataset(torch.from_numpy(X_test_scaled).to(DEVICE),
                                            torch.from_numpy(y_test).type(torch.float64).to(DEVICE))
                train_loader = DataLoader(train_dataset, batch_size=len(y_train), shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=len(y_test), shuffle=False)

                # create model
                model = MLP(arch).to(DEVICE)
                results = train_and_evaluate(model, train_loader, val_loader, 200, DEVICE)
                best_uar.append(np.max(results["uar"]))

                with open(result_dir.joinpath(f'mlp_res_{idx}.pickle'), "wb") as output_file:
                    pickle.dump(results, output_file)
            print(f"mean uar: {np.mean(best_uar)}")


if __name__ == "__main__":
    main()
