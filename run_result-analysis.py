import os
import argparse
from tqdm import tqdm

import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import smash
from smash.solver._mwd_cost import nse, kge

print("===================================")
print(f"smash version: {smash.__version__}")
print("===================================")


##################################
##  PARSER AND COMMON VARIABLES ##
##################################


def initialize_args():  # do not set new attr or modify any attr of args outside this function
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "-data", "--data", type=str, help="Select the data directory"
    )

    parser.add_argument(
        "-m", "-modeldir", "--modeldir", type=str, help="Select the model directory"
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

    args.methods = ["uniform", "distributed", "hyper-linear", "hyper-polynomial", "ann"]

    args.cal_code = pd.read_csv(os.path.join(args.data, "cal_code.csv"))[
        "cal"
    ].to_list()

    args.val_code = pd.read_csv(os.path.join(args.data, "val_code.csv"))[
        "val"
    ].to_list()

    args.models_ddt = [
        smash.read_model_ddt(os.path.join(args.modeldir, method + ".hdf5"))
        for method in tqdm(args.methods, desc="</> Reading models ddt...")
    ]

    args.cost = {"nse": nse, "kge": kge}

    return args


##############################
##  FUNCTIONS FOR ANALYSIS  ##
##############################


def boxplot_cost(args, fobj="nse", figname="boxplots", figsize=(12, 8)):
    print("</> Plotting boxplots...")

    cost = []
    metd = []

    cal_val = []
    code_catch = []

    for i, code in enumerate(args.models_ddt[0]["code"]):
        for model, method in zip(args.models_ddt, args.methods):
            qo = model["qobs"][i]
            qs = model["qsim"][i]

            if code in args.cal_code:
                cal_val.append("Cal")

            elif code in args.val_code:
                cal_val.append("Val")

            else:
                continue

            cost.append(1 - args.cost[fobj](qo, qs))

            metd.append(method)

            code_catch.append(code)

    df = pd.DataFrame(
        {"code": code_catch, "cal_val": cal_val, "Mapping": metd, "NSE": cost}
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x="cal_val",
        y="NSE",
        hue="Mapping",
        width=0.5,
        palette="deep",
        showfliers=False,
        ax=ax,
        order=["Cal", "Val"],
    )

    # Set title and axis labels
    ax.set(title=None, xlabel=None, ylabel="NSE")

    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles[0 : len(args.methods)],
        labels[0 : len(args.methods)],
        title="Mapping Type",
        title_fontsize=14,
        loc="upper right",
        bbox_to_anchor=(1.3, 1),
    )

    # Set y-axis limits and add grid
    ax.set_ylim([0, 1])
    ax.yaxis.grid(True)

    ax.xaxis.grid(False)

    # Adjust plot spacing
    plt.subplots_adjust(right=0.8)

    plt.savefig(os.path.join(args.output, figname + ".png"))


def hydrograph(
    args, cal_val="cal", figname="hydrograph_cal", figsize: tuple | None = None
):
    # figsize=None: auto adjusted figure size

    print(f"</> Plotting hydrograph ({cal_val})...")

    if cal_val.lower() == "cal":
        codes = args.cal_code

    elif cal_val.lower() == "val":
        codes = args.val_code

    else:
        raise ValueError(f"cal_val should be a str, either cal or val, not {cal_val}")

    nr = len(codes)
    nc = len(args.methods)

    if figsize == None:
        figsize = (3 * nc, nr)

    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=figsize)

    for j, mtd in enumerate(args.methods):
        axes[0, j].set_title(mtd, fontsize=14)

        i = 0

        for c, catch in enumerate(args.models_ddt[j]["code"]):
            if catch in codes:
                qo = args.models_ddt[j]["qobs"][c, :]
                qs = args.models_ddt[j]["qsim"][c, :]

                qo[np.where(qo < 0)] = np.nan
                qs[np.where(qs < 0)] = np.nan

                t = range(len(qo))

                ax = axes[i, j]
                i += 1

                ax.plot(t, qo, color="red", label="Observed")
                ax.plot(t, qs, color="blue", linestyle="-.", label="Simulated")
                ax.tick_params(axis="both", which="both", labelsize=10)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.yaxis.grid(True)
                ax.xaxis.grid(False)

                if j > 0:
                    ax.set_yticklabels([])

    # Add a legend to the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12)

    # Add axis name
    fig.text(0.5, 0.075, "Time step", ha="center", fontsize=14)
    fig.text(
        0.075, 0.5, "Discharge (m$^3$/s)", va="center", rotation="vertical", fontsize=14
    )

    plt.savefig(os.path.join(args.output, figname + ".png"))


def param_map(
    args,
    params=["exc", "lr", "cft", "cp"],
    bounds=[(-20, 5), (1, 200), (1, 1000), (2, 2000)],
    figname="param_map",
    figsize=(12, 9),
):
    print("</> Plotting parameters map...")

    fig, axes = plt.subplots(nrows=3, ncols=len(params), figsize=figsize)

    for j, par in enumerate(params):
        axes[0, j].set_title(params[j])

        for i, mod in enumerate(args.models_ddt[2:]):
            axes[i, j].yaxis.grid(False)
            axes[i, j].xaxis.grid(False)

            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            im = axes[i, j].imshow(
                mod[par],
                cmap="viridis",
                vmin=bounds[j][0],
                vmax=bounds[j][1],
                interpolation="bicubic",
                alpha=1.0,
            )

            if j == 0:
                axes[i, j].set_ylabel(args.methods[i + 2], labelpad=10)

        divider = make_axes_locatable(axes[-1, j])
        cax = divider.new_vertical(size="5%", pad=0.2, pack_start=True)
        # Add a colorbar to the new axes
        fig.add_axes(cax)
        fig.colorbar(im, cax=cax, orientation="horizontal")

    plt.savefig(os.path.join(args.output, figname + ".png"))


def cost_descent(args, niter=250, figsize=(12, 6), figname="cost_descent"):
    print("</> Plotting cost descent...")

    x = range(1, niter + 1)

    # Define colors and line styles for each method
    colors = [
        "#dd8452",
        "#55a868",
        "#c44e52",
        "#8172b3",
    ]  # corresponding colors for palette='deep'
    linestyles = ["--", ":", "-.", "-"]
    line_kwargs = {"linewidth": 2}

    fig, ax = plt.subplots(figsize=figsize)

    for i, mtd in enumerate(args.methods[1:]):
        if mtd == "ann":
            J = np.loadtxt(
                args.modeldir + "outterminal/" + mtd + ".txt", usecols=(5, 9)
            )
        else:
            J = np.loadtxt(
                args.modeldir + "outterminal/" + mtd + ".txt", usecols=(8, 12)
            )

        ax.plot(
            x,
            J[: len(x), 0],
            label=mtd,
            color=colors[i],
            linestyle=linestyles[i],
            **line_kwargs,
        )

    # Set x and y axis labels, title and legend
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Cost", fontsize=14)
    # ax.set_title("Multisites Optimization", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)

    # Customize tick labels and grid lines
    ax.tick_params(axis="both", which="major", labelsize=12, width=2, length=6)
    ax.tick_params(axis="both", which="minor", labelsize=10, width=1, length=4)
    ax.grid(
        True, which="major", axis="both", linestyle="--", color="lightgray", alpha=0.7
    )

    plt.savefig(os.path.join(args.output, figname + ".png"))


##########
## MAIN ##
##########

if __name__ == "__main__":

    # Increase font size
    sns.set(font_scale=1.2)

    args = initialize_args()

    hydrograph(args, "cal", "hydrograph_cal")
    hydrograph(args, "val", "hydrograph_val")

    boxplot_cost(args)

    param_map(args)

    # cost_descent(args, niter=262)
