"""
This is the main file of the project. It is used to run all the other files in the correct order
and to check everything is working as expected.
"""
import sys
from pathlib import Path
from multiprocessing import freeze_support

from src.checksum import test_file_list_from_file
#
# from data_preprocessing import main as data_preprocessing
# from feature_extraction import main as feature_extraction
# from datasets_generator import main as datasets_generator

from ml_pipeline import main as classifier_pipeline

# machine dependent settings
MAX_WORKERS = 24

# dir and filenames
# ORIGINAL_DATA_DIR = Path("svd_db")
# PREPROCESSED_DATA_DIR = Path("dataset")
CREATED_DATASETS_DIR = Path("training_data")
# FILE_INFORMATION_PATH = Path("misc").joinpath("svd_information.csv")
# REQUIRED_INITIAL_CHECKSUMS = Path("misc").joinpath("data_used.sha256")
# CHECKSUMS_AFTER_I    = Path(".").joinpath("misc","after_I.sha256")
# CHECKSUMS_AFTER_II   = Path(".").joinpath("misc","after_II.sha256")
# CHECKSUMS_AFTER_III  = Path(".").joinpath("misc","after_III.sha256")
# CHECKSUMS_AFTER_IV   = Path(".").joinpath("misc","after_IV.sha256")

if __name__ == "__main__":
    freeze_support()

    # check if the folder svd_db exists and if the checskums are the same
    # print("Checking if the input data is the same as ours.")
    # check_futher,_ = test_file_list_from_file(REQUIRED_INITIAL_CHECKSUMS)
    # if not check_futher:
    #     print("The initial data is not the same as the one provided in the file data_used.sha256")
    #     if input("Continue anyway? (y/n): ") != "y":
    #         sys.exit(1)
    #
    # # 1. run data_preprocessing.py
    # print("-"*79)
    # print("Step 1: Data Preprocessing")
    # print("Running data preprocessing.")
    # data_preprocessing(source_path=ORIGINAL_DATA_DIR,
    #                    destination_path=PREPROCESSED_DATA_DIR,
    #                    file_information_path=FILE_INFORMATION_PATH)
    #
    #
    #
    # # check if the folder dataset is created and if the checksums are the same
    # if check_futher:
    #     print("Checking if the output after data preprocessing is the same as ours.")
    #     ok,_ = test_file_list_from_file(CHECKSUMS_AFTER_I)
    #     if not ok:
    #         print("The output data after data preprocessing is not the same as ours!")
    #         print("Continuing anyway. Press Ctrl+C to stop.")
    #
    # # 2. run feature_extraction.py
    # print("-"*79)
    # print("Step 2: Feature Extraction")
    # print("Running feature extraction.")
    # feature_extraction(source_dictionary=PREPROCESSED_DATA_DIR,
    #                    round_digits=6)
    #
    # # check if the created files are the same as ours
    # if check_futher:
    #     print("Checking if the output after feature extraction is the same as ours.")
    #     ok,_ = test_file_list_from_file(CHECKSUMS_AFTER_II)
    #     if not ok:
    #         print("The output data after feature extraction is not the same as ours!")
    #         print("Continuing anyway. Press Ctrl+C to stop.")
    #
    #
    # # 3. run datasets_generator.py
    # print("-"*79)
    # print("Step 3: Datasets Generation")
    # print("Running datasets generation.")
    # datasets_generator(max_workers=MAX_WORKERS)

    # check if the created files are the same as ours
    # if check_futher:
    #     print("Checking if the output after datasets generation is the same as ours.")
    #     ok,_ = test_file_list_from_file(CHECKSUMS_AFTER_III)
    #     if not ok:
    #         print("The output data after datasets generation is not the same as ours!")
    #         print("Continuing anyway. Press Ctrl+C to stop.")
    #
    #
    # TODO
    # 4. run svm_pipeline.py
    sexes = ["women"]
    classifiers = ["svm_poly"]
    for sex in sexes:
        for classifier in classifiers:
            classifier_pipeline(sex=sex, classifier=classifier)
    # 5. run analyze_results.py to check the results
    # 6. run average_results.py to get the best datasets according to average performance of classifiers
    # 7. run svm_repeated_xvalidation.py to evaluate the best classifiers on the best on 10 ten average datasets

    # TODO: skip steps 1-3 if the CHECKSUMS_AFTER_III is correct
