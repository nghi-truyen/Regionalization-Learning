import os
import argparse
from tqdm import tqdm
import random

import seaborn as sns
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import smash
from smash.solver._mwd_cost import nse, kge

if smash.__version__ == "0.3.0":
    print("===================================")
    print(f"smash version: {smash.__version__}")
    print("===================================")

else:
    raise ValueError("Only support for smash 0.3.0 version")


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


def compare_cost(args, fobj="nse", figname="compare_cost", figsize=(15, 8)):
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
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # boxplot
    sns.boxplot(
        data=df,
        x="cal_val",
        y="NSE",
        hue="Mapping",
        width=0.5,
        palette="deep",
        showfliers=True,
        ax=axes[0],
        order=["Cal", "Val"],
    )

    # Set title and axis labels
    axes[0].set(title=None, xlabel=None, ylabel="NSE")

    # Set y-axis limits and add grid
    axes[0].set_ylim([0, 1])
    axes[0].yaxis.grid(True)

    axes[0].xaxis.grid(False)

    handles, labels = axes[0].get_legend_handles_labels()  # get labels then remove
    axes[0].legend([], [], frameon=False)

    # NSE by catchment
    colors = [
        "#4c72b0",
        "#dd8452",
        "#55a868",
        "#c44e52",
        "#8172b3",
    ]  # corresponding colors for palette='deep

    cls = dict(zip(args.methods, colors))

    for i, catch in enumerate(code_catch):
        if cal_val[i] == "Cal":
            axes[1].scatter(
                catch,
                cost[i],
                color=cls[metd[i]],
                marker="s",
                s=100,
                label="Cal" if i == 0 else None,
            )

    for i, catch in enumerate(code_catch):  # to separate cal and val code
        if cal_val[i] == "Val":
            axes[1].scatter(
                catch,
                cost[i],
                color=cls[metd[i]],
                marker="^",
                s=100,
                label="Val" if i == 0 else None,
            )

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker=m,
            color="None",
            label=l,
            linestyle="",
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        )
        for m, l in zip(["s", "^"], ["Cal", "Val"])
    ]
    axes[1].legend(handles=legend_elements, loc="lower left", fontsize=14)

    axes[1].xaxis.grid(False)

    axes[1].set_xlabel("Catchment", fontsize=13)

    axes[1].set_xticklabels([])

    fig.legend(
        handles,
        labels,
        title=None,
        loc="lower center",
        ncol=len(args.methods),
        fontsize=14,
    )

    plt.savefig(os.path.join(args.output, figname + ".png"))


def hydrograph(
    args, cal_val="cal", figname="hydrograph_cal", figsize: tuple | None = None
):  # figsize=None: auto adjusted figure size
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

        ind_code = [
            ind for ind, c in enumerate(args.models_ddt[j]["code"]) if c in codes
        ]

        for i, ic in enumerate(ind_code):
            qo = np.copy(args.models_ddt[j]["qobs"][ic, :])
            qs = np.copy(args.models_ddt[j]["qsim"][ic, :])

            qo[np.where(qo < 0)] = np.nan
            qs[np.where(qs < 0)] = np.nan

            t = range(len(qo))

            ax = axes[i, j]

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

            if j > 0:  # remove ticklabel
                ax.set_yticklabels([])

    # set the same scale for y-axis
    for i in range(axes.shape[0]):
        ymin = min([ax.get_ylim()[0] for ax in axes[i, :]])

        ymax = max([ax.get_ylim()[1] for ax in axes[i, :]])

        for j in range(axes.shape[1]):
            axes[i, j].set_ylim([ymin, ymax])

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
    cmaps=["RdBu", "Purples", "YlGnBu", "viridis"],
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
                cmap=cmaps[j],
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


def desc_map(
    args,
    desc=[
        "pente",
        "ddr",
        "karst2019_shyreg",
        "foret",
        "urbain",
        "resutilpot",
        "vhcapa",
    ],
    cmap="terrain",
    figname="desc_map",
    figsize=(12, 3),
):
    print("</> Plotting descriptors map...")

    descriptor = dict.fromkeys(desc, None)

    with h5py.File(os.path.join(args.data, "descriptors.hdf5"), "r") as f:
        for name in desc:
            descriptor[name] = np.copy(f[name][:])

    fig, axes = plt.subplots(nrows=1, ncols=len(desc), figsize=figsize)

    for i, darr in enumerate(descriptor.values()):
        axes[i].set_title("d" + str(i + 1), fontsize=10)

        axes[i].yaxis.grid(False)
        axes[i].xaxis.grid(False)

        axes[i].set_xticks([])
        axes[i].set_yticks([])

        im = axes[i].imshow(
            darr,
            cmap=cmap,
            interpolation="bicubic",
            alpha=1.0,
        )

        cbar = fig.colorbar(
            im, ax=axes[i], orientation="horizontal", pad=0.1, aspect=15
        )

        cbar.ax.tick_params(labelsize=8)

    plt.savefig(os.path.join(args.output, figname + ".png"))


def linear_cov(
    args,
    params=["exc", "lr", "cft", "cp"],
    desc=[
        "pente",
        "ddr",
        "karst2019_shyreg",
        "foret",
        "urbain",
        "resutilpot",
        "vhcapa",
    ],
    figname="linear_cov",
    figsize=(6, 3.5),
):
    print("</> Plotting linear covariance matrix...")

    descriptor = dict.fromkeys(desc, None)

    with h5py.File(os.path.join(args.data, "descriptors.hdf5"), "r") as f:
        for name in desc:
            descriptor[name] = np.copy(f[name][:])

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, constrained_layout=True)

    for k, mod in enumerate(args.models_ddt[2:]):
        cov_mat = np.zeros((len(desc), len(params)))

        for i, dei in enumerate(descriptor.values()):
            for j, par in enumerate(params):
                pai = np.copy(mod[par])

                # create a linear regression model
                lm = LinearRegression()

                # fit the model to the data
                lm.fit(dei.reshape(-1, 1), pai.reshape(-1, 1))

                # calculate the predicted values
                pai_pred = lm.predict(dei.reshape(-1, 1)).reshape(pai.shape)

                # calculate the total sum of squares (TSS)
                TSS = ((pai - np.mean(pai)) ** 2).sum()

                # calculate the residual sum of squares (RSS)
                RSS = ((pai - pai_pred) ** 2).sum()

                # calculate R-squared
                cov_mat[i, j] = 1 - (RSS / TSS)

        ytl = ["d" + str(num + 1) for num in range(len(desc)) if k == 0]

        axes[k].yaxis.grid(False)
        axes[k].xaxis.grid(False)

        sns.heatmap(
            cov_mat,
            xticklabels=params,
            yticklabels=ytl,
            vmin=0,
            vmax=1,
            cbar=k == 2,
            ax=axes[k],
            cmap="crest",
        )

        axes[k].set_title(args.methods[k + 2])

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

    compare_cost(args)

    param_map(args)

    # cost_descent(args, niter=262)

    desc_map(args)

    linear_cov(args)
