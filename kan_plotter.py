import pickle
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":
    pickled_results_path = Path(".", "[115_104_2]")

    for result in pickled_results_path.glob("*.pickle"):
        with open(result, "rb") as f:
            experiment_results = pickle.load(f)
        print(experiment_results.keys())
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # First subplot for train and test accuracy
        ax1.plot(experiment_results["train_loss"], label='Train loss', marker='o')
        ax1.plot(experiment_results["test_loss"], label='Test loss', marker='o')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Losses')
        ax1.set_title('Losses')
        ax1.legend()
        ax1.grid(True)

        # # Second subplot for test specificity and test recall
        # ax2.plot(experiment_results["test_specificity"], label='Test Specificity', marker='o')
        # ax2.plot(experiment_results["test_recall"], label='Test Recall', marker='o')
        # ax2.set_xlabel('Epochs')
        # ax2.set_ylabel('Metrics')
        # ax2.set_title('Test Specificity and Test Recall')
        # ax2.legend()
        # ax2.grid(True)
        #
        # # Third subplot for UAR
        # ax3.plot(uar, label='Test UAR', marker='o')
        #
        # ax3.set_xlabel('Epochs')
        # ax3.set_ylabel('Metrics')
        # ax3.set_title('Test UAR')
        # ax3.legend()
        # ax3.grid(True)
        #
        # # Fourth subplot for losses
        # ax4.plot(experiment_results["train_loss"], label='Train loss', marker='o')
        # ax4.plot(experiment_results["test_loss"], label='Test loss', marker='o')
        # ax4.set_xlabel('Epochs')
        # ax4.set_ylabel('Metrics')
        # ax4.set_title('Losses')
        # ax4.legend()
        # ax4.grid(True)
        #
        # # Adjust layout
        # plt.tight_layout()

        # str_input = input("Press something.. (x for end): ")
        # if str_input == "x":
        #     break
    # print(f"best uar: {best_uar}")

    plt.show()
