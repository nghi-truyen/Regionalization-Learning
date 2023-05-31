import smash
import pandas as pd
import os
import argparse
from tqdm import tqdm
import multiprocessing as mp
from preprocessing import load_data

if smash.__version__ >= "0.3.1":
    print("===================================")
    print(f"smash version: {smash.__version__}")
    print("===================================")

else:
    raise ValueError(
        "This code requires a minimum version of smash 0.3.1 or higher. Please update your smash installation."
    )


BOUNDS = {"cp": [2, 2000], "cft": [1, 1000], "exc": [-20, 5], "lr": [1, 200]}


parser = argparse.ArgumentParser()

parser.add_argument("-d", "-data", "--data", type=str, help="Select the data directory")

parser.add_argument(
    "-m",
    "-method",
    "--method",
    type=str,
    help="Select optimization method",
    choices=["local-uniform", "local-distributed"],
)

parser.add_argument(
    "-n",
    "-ncpu",
    "--ncpu",
    type=int,
    help="Select the number of CPU if using multiprocessing",
    default=1,
)

parser.add_argument(
    "-o",
    "-output",
    "--output",
    type=str,
    help="[optional] Set the output directory / Default: current directory",
    default=f"{os.getcwd()}",
)

args = parser.parse_args()

if not os.path.exists(os.path.join(args.output, args.method)):
    os.makedirs(os.path.join(args.output, args.method))


def local_optimize(df, start_time, end_time):
    setup, mesh = load_data(
        df,
        start_time=start_time,
        end_time=end_time,
        desc_dir="...",
    )

    model = smash.Model(setup, mesh)

    if args.method == "local-uniform":
        model.optimize(
            mapping="uniform",
            algorithm="sbs",
            bounds=BOUNDS,
            options={"maxiter": 100},
            inplace=True,
            verbose=True,
        )

    elif args.method == "local-distributed":
        model.optimize(
            mapping="uniform",
            algorithm="sbs",
            bounds=BOUNDS,
            options={"maxiter": 6},
            inplace=True,
            verbose=True,
        )

        model.optimize(
            mapping="distributed",
            algorithm="l-bfgs-b",
            bounds=BOUNDS,
            options={"maxiter": 300},
            inplace=True,
            verbose=True,
        )

    smash.save_model_ddt(
        model,
        path=os.path.join(args.output, args.method, model.mesh.code[0] + ".hdf5"),
        sub_data={"cal_cost": model.output.cost},
    )


##########
## MAIN ##
##########


df = pd.read_csv(os.path.join(args.data, "info_bv.csv"))

start_time = "2006-08-01 00:00"
end_time = "2016-08-01 00:00"

if args.ncpu > 1:
    pool = mp.Pool(args.ncpu)

    pool.starmap(
        local_optimize,
        [
            (dfi, start_time, end_time)
            for dfi in tqdm(df.iloc, desc="</> Local calibration")
        ],
    )

    pool.close()

else:
    for dfi in tqdm(df.iloc, desc="</> Local calibration"):
        local_optimize(dfi, start_time, end_time)
