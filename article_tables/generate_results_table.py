"""
This script handles generation of the latex table for the best performing datasets for each classifier and sex
"""
from pathlib import Path
import pandas as pd



SEXES = ["women", "men"]

#TODO:
BEST_MLP = {
        "women": {'mcc': 0.6940727550676049, 'sensitivity': 0.8722727272727273, 'specificity': 0.8186585365853659,
                  'gm': 0.8442666447422743, 'uar': 0.8454656319290466, 'bm': 0.6909312638580931,
                  'mcc_std': 0.09287098007890512, 'sensitivity_std': 0.05836687146779787,
                  'specificity_std': 0.056978240834672324, 'bm_std': 0.08929959941394419, 'gm_std': 0.0443615870884772,
                  'uar_std': 0.044649799706972096, 'architecture': '[126_215_126]'},
        "men": {'mcc': 0.6448697777023891, 'sensitivity': 0.8352525252525252, 'specificity': 0.8215384615384617,
                'gm': 0.82699808623235, 'uar': 0.8283954933954935, 'bm': 0.6567909867909869,
                'mcc_std': 0.08954504365184816, 'sensitivity_std': 0.03194523295229973,
                'specificity_std': 0.089463886469617, 'bm_std': 0.09677573885992274, 'gm_std': 0.04936465669621001,
                'uar_std': 0.04838786942996137, 'architecture': '[115_219_173]'}
    }

BEST_KAN = {
    "women": {'mcc': 0.7164649843741262, 'sensitivity': 0.8830194805194805, 'specificity': 0.8309146341463416,
              'gm': 0.8560258569690833, 'uar': 0.8569670573329109, 'bm': 0.713934114665822,
              'mcc_std': 0.0667263464453132, 'sensitivity_std': 0.043284481515014295,
              'specificity_std': 0.04543894092918771, 'bm_std': 0.06430668185386339, 'gm_std': 0.03212621389792792,
              'uar_std': 0.032153340926931694, 'architecture': '[126_89_2]'},
    "men": {'mcc': 0.6628184433094418, 'sensitivity': 0.8614567602939696, 'specificity': 0.804923076923077,
            'gm': 0.8317785565599813, 'uar': 0.8331899186085232, 'bm': 0.6663798372170466,
            'mcc_std': 0.1225847591513746, 'sensitivity_std': 0.05410395535716146,
            'specificity_std': 0.08731308014551452, 'bm_std': 0.12332383365464128, 'gm_std': 0.06228833741357129,
            'uar_std': 0.06166191682732064, 'architecture': '[115_173_127]'}
    }

def best_dataset_for_each_clf():
    """
    Creating a latex table from all results for a single gender to report the best performing datasets for each classifier type
    param sex: men or women, selection of what kind of results should be reported
    return result_merged: Table with both the best results for each classifier and the dataset configuration
    """
    results_path = Path(".").resolve().parents[0].joinpath("results")
    classifiers = [item.name for item in results_path.iterdir()]
    results = {"men": [],
               "women": []}
    cls_params = {"men": {},
               "women": {}}
    for classifier in classifiers:
        for sex in SEXES:
            to_process = pd.read_csv(results_path.joinpath(classifier, sex, "0", "results.csv"))

            best = to_process[to_process.mean_test_mcc == to_process.mean_test_mcc.max()].iloc[0]
            cls_params[sex][f"{classifier.upper().replace("_", " ")}"] = best.params
            best["classifier"] = classifier
            best = best.drop(["params", "mean_test_accuracy", "std_test_accuracy"])
            results[sex].append(best)

    results_tables = {"men": [],
                      "women": []}
    for sex in SEXES:
        results_tables[sex] = pd.DataFrame(results[sex])
        results_tables[sex]["classifier"] = results_tables[sex]["classifier"].apply(lambda x: x.upper().replace("_", " "))
        results_tables[sex] = results_tables[sex][["mean_test_mcc", "std_test_mcc",
                                                  "mean_test_recall", "std_test_recall",
                                                   "mean_test_specificity", "std_test_specificity",
                                                   "mean_test_uar", "std_test_uar",
                                                   "mean_test_bm", "std_test_bm",
                                                   "mean_test_gm", "std_test_gm", "classifier"]]
        print(results_tables[sex])
        results_tables[sex].set_index("classifier", inplace=True)

        # Delete SVM RBF as it is worse than SVM POLY
        results_tables[sex] = results_tables[sex].transpose().drop("SVM RBF", axis=1)
    return results_tables, cls_params

def generate_table_with_results(table):
    """
    Generating a table with all the results for the article for a single gender to report the best performing datasets for each classifier
    param table: Table with all the results for a single gender to report the best performing datasets for each classifier type
    param params: Parameters for the classifier type
    return None
    """
    table_codes = []
    for sex in SEXES:
        # Concatenate values in each row with & so it can be added to the latex table template
        print("Generating latex table...")
        # Declaration of table
        header = ("\\begin{table}\n"
                  "\\centering\n")
        # Caption
        header += f"\\caption{{Best performance for each classifier - {sex}.}}\n"
        # Declaration of tabular
        header += (f"\\begin{{tabular}}{{ll{'c' * table[sex].columns.shape[0]}}}\n"
                   f"\\toprule\n")
        # Header
        header += (f"\\multicolumn{{2}}{{l}}{{Classifier}} & {' & '.join(table[sex].columns.tolist())} \\\\\n"
                   f"\\midrule\n")

        # Body - performance metrics
        body_metrics = ""
        for i, (index, row) in enumerate(table[sex].iterrows()):
            vals = [f"{x:.4f}" for x in row]
            if i % 2 == 0:
                body_metrics += f"\\multirow{{2}}{{*}}{{{index.split("_")[-1].upper()[:3]}}} & $\\mu$ & {' & '.join(vals)} \\\\\n"
            else:
                body_metrics += f" & $\\sigma$ & {' & '.join(vals)} \\\\\n\\midrule\n"

        # Getting rid of the obsolete last \midrule
        body_metrics = body_metrics[:-9]
        # Ending tabular
        footer = ("\\bottomrule\n"
                  "\\end{tabular}\n")
        # Label
        footer += f"\\label{{tab:results_{sex}}}\n"
        # Ending table
        footer += "\\end{table}\n"
        # Saving the code to a list
        table_codes.append(header + body_metrics + footer)

    with open("results_tables.tex", "w") as f:
        f.write("\n\n".join(table_codes))

def generate_table_with_configs(table, params):
    """
       Generating a table with all the results for the article for a single gender to report the best performing datasets for each classifier
       param table: Table with all the results for a single gender to report the best performing datasets for each classifier type
       param params: Parameters for the classifier type
       return None
       """
    table_codes = []
    for sex in SEXES:
        # Concatenate values in each row with & so it can be added to the latex table template
        print("Generating latex table...")
        # Declaration of table
        header = ("\\begin{table}\n"
                  "\\centering\n")
        # Caption
        header += f"\\caption{{Configuration of he best performing classifiers - {sex}.}}\n"
        # Declaration of tabular
        header += ("\\begin{tabular}{ll}\n"
                   "\\toprule\n")
        # Header
        header += ("Classifier & Parameters\\\\\n"
                   "\\midrule\n")
        # Body - classifier settings
        body_classifier = ""
        for col in table[sex].columns:
            cell = params[sex][col].replace("{", "").replace("}", "").replace("'", "")
            cell = cell.replace("classifier__", "").replace(", ", " \\\\ ").replace("_", "\\_")
            body_classifier += (f"{col} & \\shortstack[l]{{{cell}}} \\\\\n"
                                "\\midrule\n")

        body_classifier = body_classifier[:-9]
        # Ending tabular
        footer = ("\\bottomrule\n"
                  "\\end{tabular}\n")
        # Label
        footer += f"\\label{{tab:configs_{sex}}}\n"
        # Ending table
        footer += "\\end{table}\n"
        # Saving the code to a list
        table_codes.append(header + body_classifier + footer)

    with open("configs_tables.tex", "w") as f:
        f.write("\n\n".join(table_codes))

def generate_results_from_nn(best_results, index = "MLP"):

    tables = {}
    params = {}
    for sex in SEXES:
        tables[sex] = (pd.DataFrame(best_results[sex], index=[index]))
        tables[sex].drop("architecture", axis=1, inplace=True)
        new_col_names = []
        for col in tables[sex].columns:
            if "std" in col:
                new_col_names.append(f"std_test_{col.split('_')[0].replace("sensitivity", "recall")}")
            else:
                new_col_names.append(f"mean_test_{col.replace("sensitivity", "recall")}")
        tables[sex].columns = new_col_names
        tables[sex] = tables[sex].transpose()

        params[sex] = f'{{architecture: {best_results[sex]["architecture"].replace("[", "").replace("]", "").replace("_", "-")}}}'

    return tables, params



if __name__ == "__main__":
    # Generate results from ML classifiers
    table_all, params_all = best_dataset_for_each_clf()
    # Generate results from MLP
    table_mlp, params_mlp = generate_results_from_nn(BEST_MLP)
    # Generate results from KAN
    table_kan, params_kan = generate_results_from_nn(BEST_KAN, index="KAN")
    # Merge all results to one table
    for current_sex in SEXES:
        table_all[current_sex] = table_all[current_sex].join([table_mlp[current_sex], table_kan[current_sex]])
        params_all[current_sex]["MLP"] = params_mlp[current_sex]
        params_all[current_sex]["KAN"] = params_kan[current_sex]
    # Generate tables
    generate_table_with_results(table_all)
    generate_table_with_configs(table_all, params_all)
