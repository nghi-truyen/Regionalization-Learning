import smash
import pandas as pd
import os
import time
import pickle
import argparse
from preprocessing import load_data

if smash.__version__ >= "1.0":
    print("===================================")
    print(f"smash version: {smash.__version__}")
    print("===================================")

else:
    raise ValueError(
        "This code requires a minimum version of smash 1.0 or higher. Please update your smash installation."
    )


DESC_NAME = [
    "pente",
    "ddr",
    "karst2019_shyreg",
    "foret",
    "urbain",
    "resutilpot",
    "vhcapa",
]
START = "2016-08-01"
END_WARMUP = "2017-07-31"
END = "2020-07-31"


parser = argparse.ArgumentParser()

parser.add_argument(
    "-f", "-file", "--file", type=str, help="Select catchment information file"
)

parser.add_argument(
    "-m",
    "-mapping",
    "--mapping",
    type=str,
    help="Select mapping for the optimization",
    choices=["Uniform", "Multi-linear", "Multi-polynomial", "ANN"],
)

parser.add_argument(
    "-g",
    "-gauge",
    "--gauge",
    type=str,
    help="Select gauge type for the optimization",
    choices=["upstream", "downstream", "intermediate", "independent"],
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
    args.file,
    start_time=START,
    end_time=END,
    desc_name=DESC_NAME,
)

print(f"Studied period: {setup['start_time']} - {setup['end_time']}")
print(f"Studied descriptors: {setup['descriptor_name']}")

catch_info = pd.read_csv(args.file)
cal_code = catch_info[catch_info["nature"] == args.gauge]["code"].to_list()

# %%% Create Model object %%%
print("=====================")
print("     MODEL OBJECT    ")
print("=====================")

model = smash.Model(setup, mesh)


# %%% Optimize Model %%%
print("=====================")
print("   MODEL OPTIMIZE    ")
print("=====================")

common_options = {"ncpu": args.ncpu, "verbose": True}
cost_options = {"gauge": cal_code, "wgauge": "mean", "end_warmup": END_WARMUP}
return_options = {
    "iter_cost": True,
    "iter_projg": True,
    "control_vector": True,
    "net": True,
}

ts_sim = time.time()

if args.mapping == "Uniform":
    # Define optimize options
    optimizer = "sbs"
    optimize_options = {"termination_crit": dict(maxiter=100)}

elif args.mapping in ["Multi-linear", "Multi-polynomial"]:
    # First guess
    optimize_options_fg = {"termination_crit": dict(maxiter=5)}
    model.optimize(
        mapping="uniform",
        optimizer="sbs",
        optimize_options=optimize_options_fg,
        cost_options=cost_options,
        common_options=common_options,
    )

    # Define optimize options
    optimizer = "lbfgsb"
    optimize_options = {"termination_crit": dict(maxiter=250)}

elif args.mapping == "ANN":
    # Custom Net
    net = smash.factory.Net()

    dopt = smash.default_optimize_options(model)

    nd = model.setup.nd
    cv = dopt["parameters"]
    bounds = list(dopt["bounds"].values())

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
        "net": net,
        "learning_rate": 0.003,
        "termination_crit": dict(epochs=350, early_stopping=80),
    }

ret = model.optimize(
    mapping=args.mapping,
    optimizer=optimizer,
    optimize_options=optimize_options,
    cost_options=cost_options,
    common_options=common_options,
    return_options=return_options,
)

te_sim = time.time()

print("======================")
print(f"</> Calibration time: {(te_sim - ts_sim) / 3600} hours")
print("======================")

# Save optimized model
smash.io.save_model(
    model,
    path=os.path.join(args.output, args.mapping + ".hdf5"),
)

# Save supplementary object
with open(os.path.join(args.output, args.mapping + "_ret.pickle"), "wb") as f:
    pickle.dump(ret, f)
