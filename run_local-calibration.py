import smash
import pandas as pd
import os
import argparse
from tqdm import tqdm
import multiprocessing as mp
from preprocessing import load_data

if smash.__version__ >= "1.0":
    print("===================================")
    print(f"smash version: {smash.__version__}")
    print("===================================")

else:
    raise ValueError(
        "This code requires a minimum version of smash 1.0 or higher. Please update your smash installation."
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "-f", "-file", "--file", type=str, help="Select catchment information file"
)

parser.add_argument(
    "-m",
    "-mapping",
    "--mapping",
    type=str,
    help="Select optimization mapping",
    choices=["Uniform", "Distributed"],
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

if not os.path.exists(os.path.join(args.output, args.mapping)):
    os.makedirs(os.path.join(args.output, args.mapping))


def local_optimize(df, start_time, end_time, warmup):
    setup, mesh = load_data(
        df,
        start_time=start_time,
        end_time=end_time,
        desc_dir="...",
    )

    common_options = {"verbose": True}
    cost_options = {"end_warmup": warmup}

    model = smash.Model(setup, mesh)

    if args.mapping == "Uniform":
        # Define optimize options
        optimizer = "sbs"
        optimize_options = {"termination_crit": dict(maxiter=50)}

    elif args.mapping == "Distributed":
        # First guess
        optimize_options_fg = {"termination_crit": dict(maxiter=2)}
        model.optimize(
            mapping="uniform",
            optimizer="sbs",
            optimize_options=optimize_options_fg,
            cost_options=cost_options,
            common_options=common_options,
        )

        ## Define optimize options
        optimizer = "lbfgsb"
        optimize_options = {"termination_crit": dict(maxiter=200)}

    # Model optimization
    model.optimize(
        mapping=args.mapping,
        optimizer=optimizer,
        optimize_options=optimize_options,
        cost_options=cost_options,
        common_options=common_options,
    )

    # Save optimized model
    smash.io.save_model(
        model,
        path=os.path.join(args.output, f"{args.mapping}/{model.mesh.code[0]}.hdf5"),
    )


##########
## MAIN ##
##########

START = "2016-08-01"
END_WARMUP = "2017-07-31"
END = "2020-07-31"

df = pd.read_csv(args.file)

if args.ncpu > 1:
    pool = mp.Pool(args.ncpu)

    pool.starmap(
        local_optimize,
        [
            (dfi, START, END, END_WARMUP)
            for dfi in tqdm(df.iloc, desc="</> Local calibration")
        ],
    )

    pool.close()

else:
    for dfi in tqdm(df.iloc, desc="</> Local calibration"):
        local_optimize(dfi, START, END, END_WARMUP)
