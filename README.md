# Introduction
This Git repository is dedicated to performing various **multisite regionalization learning** methods, including the use of linear/polynomial mapping and Artificial Neural Network (ANN). It also includes analysis results, such as signal analysis and model parameters analysis.

# Requirements
To use this Git repository, you need to have the following requirements installed:
- smash >= 0.3.1
- Seaborn
- Scikit-learn

# Installation
Assuming that you have installed [smash](https://github.com/DassHydro-dev/smash) (at least version 0.3.1). Then to install dependencies packages, please follow these steps:
```bash
conda activate smash
(smash) pip install seaborn scikit-learn 
```

**_Note:_** Please refer to [these instructions](https://smash.recover.inrae.fr/getting_started/index.html) to install the latest version of smash.

# Usage
Now, you can use the scripts and analysis tools in this repository to perform regionalization calibration methods and analyze the results.

### Run hyper optimization methods
To run hyper optimization methods, which include uniform mapping, full distributed mapping, hyper linear/polynomial regionalization mapping, and ANN-based regionalization mapping, use the following command:
```bash
(smash) python3 run_hyper-method.py -d data/Med-Est/ -m uniform -o models/Med-Est/
(smash) python3 run_hyper-method.py -d data/Med-Est/ -m hyper-linear -o models/Med-Est/
(smash) python3 run_hyper-method.py -d data/Med-Est/ -m hyper-polynomial -o models/Med-Est/
(smash) python3 run_hyper-method.py -d data/Med-Est/ -m ann -o models/Med-Est/
```

### Running analysis
To run analysis, use the following command:
```bash
(smash) python3 run_result-analysis.py -d data/Med-Est/ -m models/Med-Est/ -o graphs/Med-Est/
```

**_Note:_**  Please ensure that the correct paths and file names are used in the scripts and the commands above.

