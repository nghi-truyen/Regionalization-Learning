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

To run multisite calibration methods, including global optimization method with spatially uniform control vectors (regionalization at level 0), regionalization with multivariate linear/polynomial regression, and regionalization with multilayer perceptron (ANN), use the following commands:
```bash
(smash) python3 run_regionalization.py -d data/MedEst -m uniform -o models/MedEst
(smash) python3 run_regionalization.py -d data/MedEst -m multi-linear -o models/MedEst
(smash) python3 run_regionalization.py -d data/MedEst -m multi-polynomial -o models/MedEst
(smash) python3 run_regionalization.py -d data/MedEst -m ann -o models/MedEst
```

**_Note:_** If you want to run local optimization methods (mono-gauge), which include local calibration methods with spatially uniform and distributed control vectors, use the following commands:
```bash
(smash) python3 run_local-calibration.py -d data/MedEst -m local-uniform -o models/MedEst
(smash) python3 run_local-calibration.py -d data/MedEst -m local-distributed -o models/MedEst
```

To run analysis on the results, use the following command:
```bash
(smash) python3 run_result-analysis.py -d data/MedEst -m models/MedEst -o graphs/MedEst
```

**_Note:_** Please make sure to provide the correct paths and file names in the scripts and the commands mentioned above. 

# Flags

You can adjust the command parameters as needed using the available flags, such as `-d`, `-m`, etc.. Here are the usage information and descriptions of all the available flags for each script:

```bash
usage: run_regionalization.py [-h] [-d DATA]
                              [-m {uniform,multi-linear,multi-polynomial,ann}]
                              [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  -d DATA, -data DATA, --data DATA
                        Select the data directory
  -m {uniform,multi-linear,multi-polynomial,ann}, -mapping {uniform,multi-linear,multi-polynomial,ann}, --mapping {uniform,multi-linear,multi-polynomial,ann}
                        Select mapping for the optimization
  -n NCPU, -ncpu NCPU, --ncpu NCPU
                        Select the number of CPU if using multiprocessing 
  -o OUTPUT, -output OUTPUT, --output OUTPUT
                        [optional] Set the output directory / Default: current
                        directory
```

```bash
usage: run_local-calibration.py [-h] [-d DATA]
                                [-m {local-uniform,local-distributed}]
                                [-n NCPU] [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  -d DATA, -data DATA, --data DATA
                        Select the data directory
  -m {local-uniform,local-distributed}, -method {local-uniform,local-distributed}, --method {local-uniform,local-distributed}
                        Select optimization method
  -n NCPU, -ncpu NCPU, --ncpu NCPU
                        Select the number of CPU if using multiprocessing
  -o OUTPUT, -output OUTPUT, --output OUTPUT
                        [optional] Set the output directory / Default: current
                        directory
```

```bash
usage: run_result-analysis.py [-h] [-d DATA] [-m MODELDIR] [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  -d DATA, -data DATA, --data DATA
                        Select the data directory
  -m MODELDIR, -modeldir MODELDIR, --modeldir MODELDIR
                        Select the model directory
  -o OUTPUT, -output OUTPUT, --output OUTPUT
                        [optional] Set the output directory / Default: current
                        directory
```
