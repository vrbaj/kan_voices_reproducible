"""Module for loading pipelines with corresponding classifiers and hyperparameters for grid search."""
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from src.customsmote import CustomSMOTE

grids = {
    'svm_poly': {
        "classifier__C": [0.1, 0.25,  0.5, 0.75, 1, 2.5, 5, 7.5, 10,  25, 50, 75, 100, 250, 500, 750,
                          1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250,
                          4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6570, 7000, 7250, 7500, 7750,
                          8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750, 10000, 10250, 10500, 10750, 11000,
                          11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000, 13250, 13500, 13750, 14000,
                          14250, 14500, 14750, 15000, 15250, 15500, 15750, 16000, 16250, 16500, 16750, 17000,
                          17250, 17500, 17750, 18000, 18250, 18500, 18750, 19000, 19250, 19500, 19750, 20000],
        "classifier__kernel": ["poly"],
        "classifier__gamma": [0.5, 0.25, 0.1, 0.075, 0.05,  0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, "auto"],
        "classifier__degree": [2, 3, 4, 5, 6]
    },
    'svm_rbf': {
                "classifier__C": [0.1, 0.25,  0.5, 0.75, 1, 2.5, 5, 7.5, 10,  25, 50, 75, 100, 250, 500, 750,
                          1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250,
                          4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6570, 7000, 7250, 7500, 7750,
                          8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750, 10000, 10250, 10500, 10750, 11000,
                          11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000, 13250, 13500, 13750, 14000,
                          14250, 14500, 14750, 15000, 15250, 15500, 15750, 16000, 16250, 16500, 16750, 17000,
                          17250, 17500, 17750, 18000, 18250, 18500, 18750, 19000, 19250, 19500, 19750, 20000],
        "classifier__kernel": ["rbf"],
        "classifier__gamma":  [0.5, 0.25, 0.1, 0.075, 0.05,  0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, "auto"],
    },
    'knn': {
        "classifier__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37 ,39],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2]},
    'gauss_nb': {
        "classifier__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    'random_forest': {
        "classifier__n_estimators": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                                     190, 200, 210, 220, 230, 240, 250, 260, 270, 280 , 290, 300, 310, 320, 330],
        "classifier__criterion": ["gini"],
        "classifier__min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9],
        "classifier__max_features": ["sqrt"]
    },
    'adaboost': {
        "classifier__n_estimators": [10, 20, 30, 40 , 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
                                     170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 350, 500],
        "classifier__learning_rate": [0.001, 0.01, 0.1, 1, 10]
    }

}

def get_classifier(classifier_name: str, random_seed: int = 42):
    """
    Get classifier with the given name.
    :param classifier_name: str, name of the classifier
    currently supported: "svm_poly", "svm_rbf", "knn", "gauss_nb", "random_forest", "adaboost"
    :param random_seed: int, random seed
    :return: classifier
    :return: grid for grid search
    """
    if classifier_name == "svm_poly":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", SVC(max_iter=int(1e6), random_state=random_seed))
        ])
    elif classifier_name == "svm_rbf":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", SVC(max_iter=int(1e6), random_state=random_seed))
        ])
    elif classifier_name == "knn":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", KNeighborsClassifier())
        ])
    elif  classifier_name == "gauss_nb":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", GaussianNB())
        ])
    elif classifier_name == "random_forest":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", RandomForestClassifier(random_state=random_seed))
        ])
    elif classifier_name == "adaboost":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", AdaBoostClassifier(random_state=random_seed, algorithm="SAMME"))
        ])
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return pipe, grids[classifier_name]
