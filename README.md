## Required env `smash-dev`:

```bash
conda activate smash-dev
```

## Run hyper optimization methods:

```bash
python3 run_hyper-method.py -d "infoBV/Med-Est" -m uniform -o res/model_ddt/Med-Est
python3 run_hyper-method.py -d "infoBV/Med-Est" -m distributed -o res/model_ddt/Med-Est
python3 run_hyper-method.py -d "infoBV/Med-Est" -m hyper-linear -o res/model_ddt/Med-Est
python3 run_hyper-method.py -d "infoBV/Med-Est" -m hyper-polynomial -o res/model_ddt/Med-Est
python3 run_hyper-method.py -d "infoBV/Med-Est" -m ann -o res/model_ddt/Med-Est
```

## Run analysis:

```bash
python3 run_result-analysis.py -d infoBV/Med-Est/ -m res/model_ddt/Med-Est/ -o res/analysis/Med-Est/
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
```

