This Git repository is dedicated to performing various **multisite regionalization learning** methods, including the use of linear/polynomial mapping and Artificial Neural Network (ANN). It also includes analysis results, such as signal analysis and model parameters analysis.

To use this Git repository, you need to have the following requirements installed:
- smash >= 1.0
- Seaborn
- Scikit-learn

# Installation
Assuming that you have already installed [smash](https://github.com/DassHydro-dev/smash) (at least version 1.0), please follow these steps to install the required dependencies:
- Activate `smash` environment using Conda.
```bash
conda activate smash
```
- Install the additional packages using pip.
```bash
(smash) pip install seaborn scikit-learn 
```
**_Note:_**  If you haven't installed [smash](https://github.com/DassHydro-dev/smash) yet, please refer to [these instructions](https://smash.recover.inrae.fr/getting_started/index.html) for installation guidance.

# Usage
Now, you can use the scripts and analysis tools in this repository to perform regionalization calibration methods and analyze the results.

To perform multisite (using gauges located upstream in this case) calibration methods, including global optimization method with spatially uniform control vectors (regionalization at level 0), regionalization with multivariate linear/polynomial regression, and regionalization with multilayer perceptron (ANN), use the following commands:
```bash
(smash) python3 run_regionalization.py -f catchment_info.csv -g upstream -m uniform -o models
(smash) python3 run_regionalization.py -f catchment_info.csv -g upstream -m multi-linear -o models
(smash) python3 run_regionalization.py -f catchment_info.csv -g upstream -m multi-polynomial -o models
(smash) python3 run_regionalization.py -f catchment_info.csv -g upstream -m ann -o models
```

**_Note:_** If you want to run local optimization methods (mono-gauge), which include local calibration methods with spatially uniform and distributed control vectors, use the following commands:
```bash
(smash) python3 run_local-calibration.py -f catchment_info.csv -m local-uniform -o models
(smash) python3 run_local-calibration.py -f catchment_info.csv -m local-distributed -o models
```

To run analysis on the results, use the following command:
```bash
(smash) python3 run_result-analysis.py ...
```

**_Note:_** Please make sure to provide the correct paths and file names in the scripts and the commands mentioned above. 

# Flags

You can adjust the command parameters as needed using the available flags, such as `-d`, `-m`, etc.. Here are the usage information and descriptions of all the available flags for each script:

TODO...