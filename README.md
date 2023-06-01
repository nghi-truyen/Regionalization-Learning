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

### Run global or regionalization optimization methods (multi-gauge)
To run a global optimization with spatially uniform control vectors:
```bash
(smash) python3 run_global-and-regionalization.py -d data/MedEst -m uniform -o models/MedEst
```

To run regionalization calibration methods, including multivariate linear/polynomial regression and multilayer perceptron:
```bash
(smash) python3 run_global-and-regionalization.py -d data/MedEst -m hyper-linear -o models/MedEst
(smash) python3 run_global-and-regionalization.py -d data/MedEst -m hyper-polynomial -o models/MedEst
(smash) python3 run_global-and-regionalization.py -d data/MedEst -m ann -o models/MedEst
```

### Run local optimization methods (mono-gauge)
To run local calibration methods with spatially uniform/distributed control vectors:
```bash
(smash) python3 run_local-calibration.py -d data/MedEst -m local-uniform -o models/MedEst
(smash) python3 run_local-calibration.py -d data/MedEst -m local-distributed -o models/MedEst
```

### Running analysis
To run analysis, use the following command:
```bash
(smash) python3 run_result-analysis.py -d data/MedEst -m models/MedEst -o graphs/MedEst
```

**_Note:_**  Please ensure that the correct paths and file names are used in the scripts and the commands above.
