"""
KAN arch search script.
"""
import pickle
from pathlib import Path
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import SMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.base import BaseSampler

N_SEED = 42
np.random.seed(N_SEED)


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


# path to training datasets
datasets = Path("", "training_data", "men")
# select computational device -> changed to CPU as it is faster for small datasets (as SVD)
DEVICE = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(DEVICE)
print(f"The {DEVICE} will be used for the computation..")
to_evaluate = []
for dataset in datasets.iterdir():
    print(f"evaluating dataset {dataset}")
    # load dataset
    with open(dataset.joinpath("dataset_selected.pk"), "rb") as f:
        dataset_file = pickle.load(f)
    X = np.array(dataset_file["data"])
    y = np.array(dataset_file["labels"])
    # path where to store results
    results_path = Path(".", "results_mc", dataset)
    # get the number of features
    input_size = X.shape[1]
    # define MLP architecture
    mlp_archs = [
    [input_size, input_size * 2 - int(0.1 * input_size)]
    [input_size, input_size * 2 - int(0.2 * input_size)]
    [input_size, input_size * 2 - int(0.3 * input_size)]
    [input_size, input_size * 2 - int(0.4 * input_size)]
    [input_size, input_size * 2 - int(0.5 * input_size)]
    [input_size, input_size * 2 - int(0.6 * input_size)]
    [input_size, input_size * 2 - int(0.7 * input_size)]
    [input_size, input_size * 2 - int(0.8 * input_size)]
    [input_size, input_size * 2 - int(0.9 * input_size)]
    [input_size, input_size]
    [input_size, input_size - int(0.1 * input_size)]
    [input_size, input_size - int(0.2 * input_size)]
    [input_size, input_size - int(0.3 * input_size)]
    [input_size, input_size - int(0.4 * input_size)]
    [input_size, input_size - int(0.5 * input_size)]
    [input_size, input_size, input_size]
    [input_size, input_size, input_size - int(0.1 * input_size)]
    [input_size, input_size, input_size - int(0.2 * input_size)]
    [input_size, input_size, input_size - int(0.3 * input_size)]
    [input_size, input_size, input_size - int(0.4 * input_size)]
    [input_size, input_size, input_size - int(0.5 * input_size)]
    [input_size, input_size, input_size - int(0.6 * input_size)]
    [input_size, input_size, input_size - int(0.7 * input_size)]
    [input_size, input_size, input_size - int(0.8 * input_size)]
    [input_size, input_size, input_size - int(0.9 * input_size)]]
    # iterate over KAN architectures and train for each dataset
    for arch in mlp_archs:

        # create results directory for each dataset and evaluated architecture
        result_dir = results_path.joinpath(str(arch).replace(",", "_").replace(" ", ""))
        result_dir.mkdir(parents=True, exist_ok=True)
        # Monte Carlo cross-validation = split train/test 10 times
        print(f"evaluating {str(arch)}")

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=32)
        idx = 0
        scores = []
        for train_index, test_index in skf.split(X, y):
            idx += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # KMeansSMOTE resampling. if fails 10x SMOTE resampling
            X_resampled, y_resampled = CustomSMOTE(kmeans_args={"random_state": N_SEED}).fit_resample(X_train, y_train)
            # MinMaxScaling
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train_scaled = scaler.fit_transform(X_resampled)
            X_test_scaled = scaler.transform(X_test)
            print(np.isnan(np.min(X_train_scaled)), np.isnan(np.min(X_test_scaled)))

            # feature dimension sanity check
            # print(dataset["train_input"].dtype)
            # create mlp model
            model = MLPClassifier(hidden_layer_sizes=list(arch), max_iter=30000000, random_state=42, activation="tanh",
                                  solver="lbfgs", early_stopping=True)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            uar = recall_score(y_test, y_pred, average='macro')
            scores.append(uar)
            print(scores)
            #
            with open(result_dir.joinpath(f'mlp_res_{idx}.pickle'), "wb") as output_file:
                pickle.dump(results, output_file)
