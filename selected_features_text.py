# Generate feature sets with feature analysis but without diff_pitch and NaN
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif

np.random.seed(42)

table = pd.read_csv("features.csv")
col_names = []
for col, val in zip(table.loc[0].index, table.loc[0].values):
    if str(val).count(",") > 0:
        for i in range(1, str(val).count(",") + 2):
            col_names.append(f"{col}{i}")
    else:
        col_names.append(col)

table_mu = pd.DataFrame(data=[], columns=col_names, index=[])

for index, row in table.iterrows():
    table_row = []
    for val in row:
        if isinstance(val, str):
            table_row += val.replace("[", "").replace("]", "").split(", ")
        else:
            table_row.append(val)
    table_mu.loc[index] = table_row

table_mu.set_index("session_id", inplace=True)

for sex in [0, 1]:
    data = table_mu.copy()
    data = data[data.sex == sex]
    X = data.drop(["pathology", "sex"], axis=1)
    y = data["pathology"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    print(f"X.shape {X.shape}")
    mut_info = mutual_info_classif(X, y, random_state=1)
    to_drop = np.where(mut_info == 0.)[0]
    print(X.shape[1] - len(data.columns[to_drop]))
    print(data.columns[to_drop])