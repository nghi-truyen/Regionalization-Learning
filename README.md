## Requirements:

```bash
conda activate smash-dev
pip install seaborn scikit-learn
```

## Run hyper optimization methods:

```bash
python3 run_hyper-method.py -d data/Med-Est/ -m uniform -o models/Med-Est/
python3 run_hyper-method.py -d data/Med-Est/ -m distributed -o models/Med-Est/
python3 run_hyper-method.py -d data/Med-Est/ -m hyper-linear -o models/Med-Est/
python3 run_hyper-method.py -d data/Med-Est/ -m hyper-polynomial -o models/Med-Est/
python3 run_hyper-method.py -d data/Med-Est/ -m ann -o models/Med-Est/
```

## Run analysis:

```bash
python3 run_result-analysis.py -d data/Med-Est/ -m models/Med-Est/ -o graphs/Med-Est/
```
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
```

