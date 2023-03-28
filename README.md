# Introduction
This Git repository is dedicated to performing various **multisite regionalization learning** methods, including the use of linear/polynomial mapping and Artificial Neural Network (ANN). It also includes analysis results, such as signal analysis and model parameters analysis.

# Requirements
To use this Git repository, you need to have the following requirements installed:
- smash-dev version 0.3.0
- Seaborn
- Scikit-learn

# Installation
To install the development version 0.3.0 of smash and the dependencies packages, please follow these steps:
```bash
git clone https://gitlab.irstea.fr/hydrology/smash.git
cd smash
git checkout v0.3.0
conda env create -f environment-dev.yml
conda activate smash-dev
(smash-dev) make
(smash-dev) pip install seaborn scikit-learn
```

# Usage
After completing the installation steps and returning to this Git repository, you can use the scripts and analysis tools in this repository to perform regionalization calibration methods and analyze the results.

### Run hyper optimization methods
To run hyper optimization methods, which include uniform mapping, full distributed mapping, hyper linear/polynomial regionalization mapping, and ANN-based regionalization mapping, use the following command:
```bash
(smash-dev) python3 run_hyper-method.py -d data/Med-Est/ -m uniform -o models/Med-Est/
(smash-dev) python3 run_hyper-method.py -d data/Med-Est/ -m distributed -o models/Med-Est/
(smash-dev) python3 run_hyper-method.py -d data/Med-Est/ -m hyper-linear -o models/Med-Est/
(smash-dev) python3 run_hyper-method.py -d data/Med-Est/ -m hyper-polynomial -o models/Med-Est/
(smash-dev) python3 run_hyper-method.py -d data/Med-Est/ -m ann -o models/Med-Est/
```

### Running analysis
To run analysis, use the following command:
```bash
(smash-dev) python3 run_result-analysis.py -d data/Med-Est/ -m models/Med-Est/ -o graphs/Med-Est/
```
You should see a progress message like this:
```
===================================
smash version: 0.3.0
===================================
</> Reading models ddt...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.65it/s]
...
```

> **_Note:_**  Please ensure that the correct paths and file names are used in the scripts and the commands above.

