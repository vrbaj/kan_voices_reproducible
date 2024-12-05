import random
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

THRESHOLD = 1e-5

def main(dataset_path: Path):
    data = np.load(dataset_path.joinpath("datasets.npz"))
    X_train=data['X_train']
    y_train=data['y_train']
    X_test=data['X_test']
    y_test=data['y_test']
    X_val=data['X_val']
    y_val=data['y_val']

    mut_info = mutual_info_classif(X_train, y_train, random_state=RANDOM_SEED)
    print("Train shape: ", X_train.shape)
    to_del =  np.where(mut_info <= THRESHOLD)[0]
    print("We will eliminate ", len(to_del), " features")
    X_train = np.delete(X_train, to_del, axis=1)
    X_test = np.delete(X_test, to_del, axis=1)
    X_val= np.delete(X_val, to_del, axis=1)
    print("We are left with train shape: ", X_train.shape)
    np.savez(dataset_path.joinpath("datasets_selected.npz"),
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test,
             X_val=X_val, y_val=y_val,)

    # save mutal information
    with dataset_path.joinpath("featurenames.txt").open("r") as f:
        header = f.readline().split(", ")
    with dataset_path.joinpath("mutal_information.txt").open("w") as f:
        for head,val in zip(header,mut_info.tolist()):
            f.write(f"{head},{val > THRESHOLD},{val}\n")

if __name__=="__main__":
    for sex in ["women","men"]:
        print("-"*64)
        print("Analyzing ", sex)
        dataset = Path("", "training_data", sex)
        main(dataset_path=dataset)
