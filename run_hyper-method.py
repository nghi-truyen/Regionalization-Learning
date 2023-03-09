import smash
import numpy as np
import os
import argparse
from preprocessing import load_data

print("===================================")
print(f"smash version: {smash.__version__}")
print("===================================")


DESC_NAME = [
    "pente",
    "ddr",
    "karst2019_shyreg",
    "foret",
    "urbain",
    "resutilpot",
    "vhcapa",
    # "medcapa",
]

BOUNDS = [(2, 2000), (1, 1000), (-20, 5), (1, 200)]


parser = argparse.ArgumentParser()

parser.add_argument("-d", "-data", "--data", type=str, help="Select the data directory")

parser.add_argument(
    "-m",
    "-method",
    "--method",
    type=str,
    help="Select optimization method",
    choices=["uniform", "distributed", "hyper-linear", "hyper-polynomial", "ann"],
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
    os.path.join(args.data, "info_bv.csv"), descriptor_name=DESC_NAME
)

print(f"Studied period: {setup['start_time']} - {setup['end_time']}")
print(f"Studied descriptors: {setup['descriptor_name']}")

cal_code = [
    "Y4624010",
    "Y6434005",
    "Y5615030",
    "Y5325010",
    "Y5032010",
    "Y5202010",
    "Y4615020",
    "Y5424010",
    "Y5615010",
]

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

if args.method == "uniform":
    model.optimize(
        mapping="uniform",
        algorithm="sbs",
        gauge=cal_code,
        wgauge="mean",
        bounds=BOUNDS,
        inplace=True,
        verbose=True,
    )

elif args.method in ["distributed", "hyper-linear", "hyper-polynomial"]:
    model.optimize(
        mapping="uniform",
        algorithm="sbs",
        gauge=cal_code,
        wgauge="mean",
        bounds=BOUNDS,
        options={"maxiter": 6},
        inplace=True,
        verbose=True,
    )

    model.optimize(
        mapping=args.method,
        algorithm="l-bfgs-b",
        gauge=cal_code,
        wgauge="mean",
        bounds=BOUNDS,
        options={"maxiter": 300},
        inplace=True,
        verbose=True,
    )

elif args.method == "ann":
    net = smash.Net()

    nd = model.input_data.descriptor.shape[-1]

    ncv = len(BOUNDS)

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
        options={"neurons": ncv, "kernel_initializer": "glorot_uniform"},
    )
    net.add(layer="activation", options={"name": "sigmoid"})

    net.add(
        layer="scale",
        options={"bounds": BOUNDS},
    )

    net.compile(optimizer="Adam", learning_rate=0.005)

    print(net)

    model.ann_optimize(
        net=net,
        epochs=600,
        early_stopping=True,
        gauge=cal_code,
        wgauge="mean",
        bounds=BOUNDS,
        inplace=True,
        verbose=True,
    )

    np.savetxt(os.path.join(args.output, "ann_loss.out"), net.history["loss_train"])

smash.save_model_ddt(
    model,
    path=os.path.join(args.output, args.method + ".hdf5"),
    sub_data={"cal_cost": model.output.cost},
)
