# Kolmogorov Arnold Network-Based Voice Pathology Detection
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13771573.svg)](https://doi.org/10.5281/zenodo.13771573)

This repository contains the code for the paper "Kolmogorov Arnold Network-Based Voice Pathology Detection"
by Jan Vrba, Jakub Steinbach, Tomáš Jirsa, Noriyasu Homma, Yuwen Zeng, Kei Ichiji, Lukáš Hájek

## Requirements
For running experiments
- prepared dataset (see below)
- reproducing the results is computationally demanding, thus we suggest to use as many CPUs as possible

Used libraries and software
- Python 3.12.4
- see requirements.txt for all dependencies

Used setup for experiments
- AMD Ryzen 5900X 12-cores CPU
- 128 GB RAM
- 2 TB SSD disk
- Ubuntu 22.04 LTS

## Notes on reproducibility



## Dataset preparation
The dataset is not included in this repository due to the license reason, but it can be downloaded from publicly
available website. Firstly download the Saarbruecken Voice Database (SVD)
[available here](https://stimmdb.coli.uni-saarland.de/help_en.php4). You need to download all recordings of /a/ vowel
produced at normal pitch that are encoded as wav files. Due to the limitation of SVD interface, download male recordings
and female recordings separately. Then create the `svd_db` folder in the root of this project and put all recordings
there.
At this step we assume following folder structure:
```
svm_voices_research
└───misc
└───src
└───svd_db
    │   1-a_n.wav
    │   2-a_n.wav
    │   ...
    │   2610-a_n.wav
```

We provide the `svd_information.csv` file that contains the information about the SVD database (age, sex, pathologies, etc.). The file is stored in the `misc` folder and contains data scraped from the SVD website.


## Running the repository



## Description of files in this repository:

- **requirements.txt** - list of used packages
- **feature_extraction.py** - script that extracts features from preprocessed data
- **README.md** - this file
- **repeated_cross_validation_best.py** - script that performs repeated stratified cross-validation of the best classifiers
- **src** - folder with additional code
    - **custom_metrics.py** - script with custom metrics used in the classifiers
    - **custom_smote.py** - script with custom SMOTE implementation
- **misc** - folder with additional files
    - **data_used.sha256** - checksum of the downloaded data
    - **list_of_excluded_files.csv** - list of excluded files from the SVD database with the reason of exclusion
    - **svd_information.csv** - information about the SVD database
