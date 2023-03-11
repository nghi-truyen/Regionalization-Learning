# Introduction
This git repository is dedicated to performing various **_multisite regionalization learning_** methods, including the use of linear/polynomial mapping and Artificial Neural Network (ANN). It also includes analysis results, such as signal analysis and model parameters analysis.

# Requirements
To use this git repository, you need to have the following requirements installed:
- smash version 0.3.0
- Seaborn
- Scikit-learn

# Installation
To install the version 0.3.0 of smash and the dependencies packages, please follow these steps:
```bash
git checkout v0.3.0
conda activate smash-dev
make clean
make
pip install seaborn scikit-learn
```

# Usage
After completing the installation steps, you can use the scripts and analysis tools in this repository to perform regionalization calibration methods and analyze the results.

### Run hyper optimization methods
To run hyper optimization methods, which include uniform mapping, full distributed mapping, hyper linear/polynomial regionalization mapping, and ANN-based regionalization mapping, use the following command:
```bash
python3 run_hyper-method.py -d data/Med-Est/ -m uniform -o models/Med-Est/
python3 run_hyper-method.py -d data/Med-Est/ -m distributed -o models/Med-Est/
python3 run_hyper-method.py -d data/Med-Est/ -m hyper-linear -o models/Med-Est/
python3 run_hyper-method.py -d data/Med-Est/ -m hyper-polynomial -o models/Med-Est/
python3 run_hyper-method.py -d data/Med-Est/ -m ann -o models/Med-Est/
```

### Running analysis
To run analysis, use the following command:
```bash
python3 run_result-analysis.py -d data/Med-Est/ -m models/Med-Est/ -o graphs/Med-Est/
```
You should see a progress message like this:
```
===================================
smash version: 0.3.0
===================================
</> Reading models ddt...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.65it/s]
</> Plotting hydrograph (cal)...
</> Plotting hydrograph (val)...
</> Plotting boxplots...
</> Plotting parameters map...
</> Plotting descriptors map...
</> Plotting linear covariance matrix...
</> Plotting relative error of signatures...
```

> **_Note:_**  Please ensure that the correct paths and file names are used in the scripts and the commands above.

