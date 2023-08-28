import smash
import pandas as pd
import os
import argparse
from preprocessing import load_data

# if smash.__version__ >= "1.0":
#     print("===================================")
#     print(f"smash version: {smash.__version__}")
#     print("===================================")

# else:
#     raise ValueError(
#         "This code requires a minimum version of smash 1.0 or higher. Please update your smash installation."
#     )


DESC_NAME = [
    "pente",
    "ddr",
    "karst2019_shyreg",
    "foret",
    "urbain",
    "resutilpot",
    "vhcapa",
]

BOUNDS = {"cp": [2, 2000], "ct": [1, 1000], "kexc": [-20, 5], "llr": [1, 200]}


parser = argparse.ArgumentParser()

parser.add_argument("-d", "-data", "--data", type=str, help="Select the data directory")

parser.add_argument(
    "-m",
    "-mapping",
    "--mapping",
    type=str,
    help="Select mapping for the optimization",
    choices=["uniform", "multi-linear", "multi-polynomial", "ann"],
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

if not os.path.exists(args.output):
    os.makedirs(args.output)

# %%% 	Preprocess data %%%
print("===========================")
print("  GENERATE MESH AND SETUP  ")
print("===========================")

setup, mesh = load_data(
    os.path.join(args.data, "info_bv.csv"),
    structure="gr4-lr",
    start_time="2006-08-01 00:00",
    end_time="2016-08-01 00:00",
    desc_name=DESC_NAME,
)

print(f"Studied period: {setup['start_time']} - {setup['end_time']}")
print(f"Studied descriptors: {setup['descriptor_name']}")

cal_code = pd.read_csv(os.path.join(args.data, "cal_code.csv"))["cal"].to_list()

# %%% Create Model object %%%
print("=====================")
print("     MODEL OBJECT    ")
print("=====================")

model = smash.Model(setup, mesh)
print(model)


# %%% Optimize Model %%%
print("=====================")
print("   MODEL OPTIMIZE    ")
print("=====================")

common_options = {"ncpu": args.ncpu, "verbose": True}

cost_options = {"gauge": cal_code, "wgauge": "mean"}

if args.mapping == "uniform":
    ## Define optimize options
    optimizer = "sbs"
    optimize_options = {
        "parameters": list(BOUNDS.keys()),
        "bounds": BOUNDS,
        "termination_crit": dict(maxiter=100),
    }

elif args.mapping in ["multi-linear", "multi-polynomial"]:
    ## Define optimize options
    optimizer = "lbfgsb"
    optimize_options = {
        "parameters": list(BOUNDS.keys()),
        "bounds": BOUNDS,
        "termination_crit": dict(maxiter=300),
    }

    ## First guess
    optimize_options_fg = optimize_options = {
        "parameters": list(BOUNDS.keys()),
        "bounds": BOUNDS,
        "termination_crit": dict(maxiter=6),
    }
    model.optimize(
        mapping="uniform",
        optimizer="sbs",
        optimize_options=optimize_options_fg,
        cost_options=cost_options,
        common_options=common_options,
    )

elif args.mapping == "ann":
    ## Custome Net
    net = smash.factory.Net()

    nd = model.setup.nd
    cv = list(BOUNDS.keys())
    bounds = list(BOUNDS.values())

    net.add(
        layer="dense",
        options={
            "input_shape": (nd,),
            "neurons": 96,
            "kernel_initializer": "glorot_uniform",
        },
    )
    net.add(layer="activation", options={"name": "relu"})

    net.add(
        layer="dense",
        options={
            "neurons": 48,
            "kernel_initializer": "glorot_uniform",
        },
    )
    net.add(layer="activation", options={"name": "relu"})

    net.add(
        layer="dense",
        options={
            "neurons": 16,
            "kernel_initializer": "glorot_uniform",
        },
    )
    net.add(layer="activation", options={"name": "relu"})

    net.add(
        layer="dense",
        options={"neurons": len(cv), "kernel_initializer": "glorot_uniform"},
    )
    net.add(layer="activation", options={"name": "sigmoid"})

    net.add(
        layer="scale",
        options={"bounds": bounds},
    )

    ## Define optimize options
    optimizer = "adam"
    optimize_options = {
        "parameters": list(BOUNDS.keys()),
        "bounds": BOUNDS,
        "net": net,
        "learning_rate": 0.005,
        "termination_crit": dict(epochs=500),
    }

model.optimize(
    mapping=args.mapping,
    optimizer=optimizer,
    optimize_options=optimize_options,
    cost_options=cost_options,
    common_options=common_options,
)

# smash.save_model_ddt(
#     model,
#     path=os.path.join(args.output, args.mapping + ".hdf5"),
#     sub_data={"cal_cost": model.output.cost},
# )
