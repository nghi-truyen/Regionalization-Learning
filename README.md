This Git repository is dedicated to performing various **multisite regionalization learning** methods, including the use of linear/polynomial mapping and Artificial Neural Network (ANN). It also includes analysis results, such as signal analysis and model parameters analysis.

**Related paper:**  
Huynh, N. N. T., Garambois, P. A., Colleoni, F., Renard, B., Roux, H., Demargne, J., Jay‐Allemand, M., & Javelle, P. (2024). *Learning regionalization using accurate spatial cost gradients within a differentiable high‐resolution hydrological model: Application to the French Mediterranean region*. **Water Resources Research, 60**(11), e2024WR037544. [https://doi.org/10.1029/2024WR037544](https://doi.org/10.1029/2024WR037544)

To use this Git repository, you need to have the following requirements installed:
- smash >= 1.0, < 1.1
- Seaborn
- Scikit-learn

# Installation
- Install [`smash`](https://github.com/DassHydro-dev/smash) (version 1.0.0 is recommended) using pip:
```bash
pip install hydro-smash==1.0.0
```
- Install additional packages:
```bash
pip install seaborn scikit-learn 
```

# Usage
Now, you can use the scripts and analysis tools in this repository to perform regionalization calibration methods and analyze the results.

To perform multisite (using gauges located upstream in this case) calibration methods, including global optimization method with spatially uniform control vectors (regionalization at level 0), regionalization with multivariate linear/polynomial regression, and regionalization with multilayer perceptron (ANN), use the following commands:
```bash
python3 run_regionalization.py -f catchment_info.csv -g upstream -m Uniform -o models/reg-upstream
python3 run_regionalization.py -f catchment_info.csv -g upstream -m Multi-linear -o models/reg-upstream
python3 run_regionalization.py -f catchment_info.csv -g upstream -m Multi-polynomial -o models/reg-upstream
python3 run_regionalization.py -f catchment_info.csv -g upstream -m ANN -o models/reg-upstream
```

**_Note:_** If you want to run local optimization methods (mono-gauge), which include local calibration methods with spatially uniform and distributed control vectors, use the following commands:
```bash
python3 run_local-calibration.py -f catchment_info.csv -m Uniform -o models/local
python3 run_local-calibration.py -f catchment_info.csv -m Distributed -o models/local
```

To run analysis on the results, you will need additional files extracted from the model hdf5 file. 
You may refer to the notebook file for more details. Then, use the following command:
```bash
python3 run_result-analysis.py -m models/reg-upstream -g upstream -o figs
```

**_Note:_** Please make sure to provide the correct paths and file names in the scripts and the commands mentioned above. 

# Flags

You can adjust the command parameters as needed using the available flags, such as `-d`, `-m`, etc.. Here are the usage information and descriptions of all the available flags for each script:

```bash
usage: run_regionalization.py [-h] [-f FILE] [-m {Uniform,Multi-linear,Multi-polynomial,ANN}]
                              [-g {upstream,downstream,intermediate,independent}] [-n NCPU] [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  -f FILE, -file FILE, --file FILE
                        Select catchment information file
  -m {Uniform,Multi-linear,Multi-polynomial,ANN}, -mapping {Uniform,Multi-linear,Multi-polynomial,ANN}, --mapping {Uniform,Multi-linear,Multi-polynomial,ANN}
                        Select mapping for the optimization
  -g {upstream,downstream,intermediate,independent}, -gauge {upstream,downstream,intermediate,independent}, --gauge {upstream,downstream,intermediate,independent}
                        Select gauge type for the optimization
  -n NCPU, -ncpu NCPU, --ncpu NCPU
                        Select the number of CPU if using multiprocessing
  -o OUTPUT, -output OUTPUT, --output OUTPUT
                        [optional] Set the output directory / Default: current directory
```

```bash
usage: run_local-calibration.py [-h] [-f FILE] [-m {Uniform,Distributed}] [-n NCPU] [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  -f FILE, -file FILE, --file FILE
                        Select catchment information file
  -m {Uniform,Distributed}, -mapping {Uniform,Distributed}, --mapping {Uniform,Distributed}
                        Select optimization mapping
  -n NCPU, -ncpu NCPU, --ncpu NCPU
                        Select the number of CPU if using multiprocessing
  -o OUTPUT, -output OUTPUT, --output OUTPUT
                        [optional] Set the output directory / Default: current directory
```

```bash
usage: run_result-analysis.py [-h] [-m MODELDIR] [-g {upstream,downstream,intermediate,independent}] [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  -m MODELDIR, -modeldir MODELDIR, --modeldir MODELDIR
                        Select the model directory
  -g {upstream,downstream,intermediate,independent}, -gauge {upstream,downstream,intermediate,independent}, --gauge {upstream,downstream,intermediate,independent}
                        Select gauge type that has been used in the calibration process
  -o OUTPUT, -output OUTPUT, --output OUTPUT
                        [optional] Set the output directory / Default: current directory
```
