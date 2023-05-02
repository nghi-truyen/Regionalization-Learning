import os
import argparse
from tqdm import tqdm

import seaborn as sns
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from preprocessing import load_data

import smash
from smash.solver._mwd_cost import nse, kge

if smash.__version__ >= "0.3.1":
    print("===================================")
    print(f"smash version: {smash.__version__}")
    print("===================================")

else:
    raise ValueError(
        "This code requires a minimum version of smash 0.3.1 or higher. Please update your smash installation."
    )


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

    args.cal_code = pd.read_csv(os.path.join(args.data, "cal_code.csv"))[
        "cal"
    ].to_list()

    args.val_code = pd.read_csv(os.path.join(args.data, "val_code.csv"))[
        "val"
    ].to_list()

    args.methods = [
        "Uniform",
        "Multi-linear",
        "Multi-polynomial",
        "ANN",
    ]
    args.filename_method = [
        "uniform",
        "hyper-linear",
        "hyper-polynomial",
        "ann",
    ]

    args.models_ddt = [
        smash.read_model_ddt(os.path.join(args.modeldir, method + ".hdf5"))
        for method in tqdm(args.filename_method, desc="</> Reading models ddt...")
    ]

    args.cost = {"NSE": nse, "KGE": kge}

    return args


##############################
##  FUNCTIONS FOR ANALYSIS  ##
##############################


def radialplot(args, fobj="NSE", figname="radialplots", figsize=(12, 6)):
    print("</> Plotting radialplots...")

    cost_cal = {mtd: [] for mtd in args.methods}
    cost_sval = {mtd: [] for mtd in args.methods}
    code_cal = []
    code_sval = []

    for i, code in enumerate(args.models_ddt[0]["code"]):
        for model, method in zip(args.models_ddt, args.methods):
            qo = model["qobs"][i]
            qs = model["qsim"][i]

            if code in args.cal_code:
                code_cal.append(code) if code not in code_cal else code_cal
                cost_cal[method].append(1 - args.cost[fobj](qo, qs))

            elif code in args.val_code:
                code_sval.append(code) if code not in code_sval else code_sval
                cost_sval[method].append(1 - args.cost[fobj](qo, qs))

    colors = [
        "#4c72b0",
        "#dd8452",
        "#55a868",
        "#c44e52",
    ]  # corresponding colors for palette='deep'

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=figsize, subplot_kw={"projection": "polar"}
    )

    ## CAL ##

    indsort_cal = np.argsort(cost_cal["ANN"])
    rad = np.linspace(0, 2 * np.pi, len(indsort_cal), endpoint=False)

    for mtd, color in zip(args.methods, colors):
        ax1.plot(
            rad,
            np.array(cost_cal[mtd])[indsort_cal],
            linewidth=2,
            linestyle="solid",
            c=color,
        )

    # set the labels for each point on the plot
    ax1.set_xticks(rad)
    ax1.set_xticklabels(np.array(code_cal)[indsort_cal])

    ax1.set_rticks([0, 0.2, 0.4, 0.6, 0.8])  # Less radial ticks
    ax1.set_rlabel_position(-25)  # Move radial labels away from plotted line
    ax1.tick_params(axis="both", labelsize=10)  # Custom font size and scale values
    ax1.spines["polar"].set_visible(
        False
    )  # Hide the border of the circle behind the text

    ax1.set_rmin(-0.25)
    ax1.set_rmax(1)

    ax1.set_title("Cal", va="bottom")

    # Add radial label
    label_position = ax1.get_rlabel_position()
    ax1.text(
        np.radians(label_position - 5),
        (ax1.get_rmax() + ax1.get_rmin()) / 2,
        "NSE",
        rotation=label_position,
        ha="center",
        va="center",
        size=10,
    )

    ## SPATIAL VAL ##

    indsort_sval = np.argsort(cost_sval["ANN"])
    rad = np.linspace(0, 2 * np.pi, len(indsort_sval), endpoint=False)

    for mtd, color in zip(args.methods, colors):
        ax2.plot(
            rad,
            np.array(cost_sval[mtd])[indsort_sval],
            linewidth=2,
            linestyle="solid",
            c=color,
            label=mtd,
        )

    # set the labels for each point on the plot
    ax2.set_xticks(rad)
    ax2.set_xticklabels(np.array(code_sval)[indsort_sval])

    ax2.set_rticks([0, 0.2, 0.4, 0.6, 0.8])  # Less radial ticks
    ax2.set_rlabel_position(-30)  # Move radial labels away from plotted line
    ax2.tick_params(axis="both", labelsize=10)  # Custom font size and scale values
    ax2.spines["polar"].set_visible(
        False
    )  # Hide the border of the circle behind the text

    ax2.set_rmin(-0.25)
    ax2.set_rmax(1)

    ax2.set_title("Spatial Val", va="bottom")

    # Add radial label
    label_position = ax2.get_rlabel_position()
    ax2.text(
        np.radians(label_position - 5),
        (ax2.get_rmax() + ax2.get_rmin()) / 2,
        "NSE",
        rotation=label_position,
        ha="center",
        va="center",
        size=10,
    )

    # adjust the subplots layout
    fig.subplots_adjust(wspace=0.35)

    fig.legend(loc="lower center", ncols=4, fontsize=12)

    plt.savefig(os.path.join(args.output, figname + ".png"))


def boxplot_and_scatterplot(
    args, fobj="NSE", figname="box-scatterplots", figsize=(15, 8)
):
    print("</> Plotting boxplots and scatterplots...")

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
                cal_val.append("Spatial Val")

            else:
                continue

            cost.append(1 - args.cost[fobj](qo, qs))

            metd.append(method)

            code_catch.append(code)

    df = pd.DataFrame(
        {"code": code_catch, "cal_val": cal_val, "Mapping": metd, fobj: cost}
    )

    # Create the plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # boxplot
    sns.boxplot(
        data=df,
        x="cal_val",
        y=fobj,
        hue="Mapping",
        width=0.5,
        palette="deep",
        showfliers=True,
        ax=axes[0],
        order=["Cal", "Spatial Val"],
    )

    # Set title and axis labels
    axes[0].set(title=None, xlabel=None, ylabel=fobj)

    # Set y-axis limits
    axes[0].set_ylim([-0.25, 1])

    handles, labels = axes[0].get_legend_handles_labels()  # get labels then remove
    axes[0].legend([], [], frameon=False)

    colors = [
        "#4c72b0",
        "#dd8452",
        "#55a868",
        "#c44e52",
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
        if cal_val[i] == "Spatial Val":
            axes[1].scatter(
                catch,
                cost[i],
                color=cls[metd[i]],
                marker="^",
                s=100,
                label="Spatial Val" if i == 0 else None,
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
        for m, l in zip(["s", "^"], ["Cal", "Spatial Val"])
    ]
    axes[1].legend(handles=legend_elements, loc="lower left", ncols=2, fontsize=14)

    axes[1].xaxis.set_major_locator(plt.FixedLocator(axes[1].get_xticks()))
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=75, fontsize=10)

    axes[1].set_ylim(axes[0].get_ylim())

    fig.legend(
        handles,
        labels,
        title=None,
        loc="upper center",
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

        for i, mod in enumerate(args.models_ddt[1:]):
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
                axes[i, j].set_ylabel(args.methods[i + 1], labelpad=10)

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
    ]  # corresponding colors for palette='deep'
    linestyles = ["-", ":", "-."]
    line_kwargs = {"linewidth": 2}

    fig, ax = plt.subplots(figsize=figsize)

    for i, mtd in enumerate(args.filename_method[1:]):
        if mtd == "ann":
            J = np.loadtxt(
                os.path.join(args.modeldir, "outterminal/" + mtd + ".txt"),
                usecols=(5, 9),
            )
        else:
            J = np.loadtxt(
                os.path.join(args.modeldir, "outterminal/" + mtd + ".txt"),
                usecols=(8, 12),
            )

        ax.plot(
            x,
            J[: len(x), 0],
            label=args.methods[i + 1],  # without uniform method
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
    figsize=(4, 6),
):
    print("</> Plotting linear covariance matrix...")

    descriptor = dict.fromkeys(desc, None)

    with h5py.File(os.path.join(args.data, "descriptors.hdf5"), "r") as f:
        for name in desc:
            descriptor[name] = np.copy(f[name][:])

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, constrained_layout=True)

    for k, mod in enumerate(args.models_ddt[1:]):
        cov_mat = np.zeros((len(params), len(desc)))

        for j, dei in enumerate(descriptor.values()):
            for i, par in enumerate(params):
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
            xticklabels=ytl,
            yticklabels=params,
            vmin=0,
            vmax=1,
            square=True,
            cbar=k == 2,
            cbar_kws=dict(
                use_gridspec=False, location="bottom", shrink=0.75, aspect=40
            ),
            ax=axes[k],
            cmap="crest",
        )

        axes[k].tick_params(
            labelright=True,
            labelleft=False,
            labelbottom=False,
            labeltop=True,
            labelrotation=0,
        )

        axes[k].set_ylabel(args.methods[k + 1], fontsize=13, labelpad=10)

    plt.savefig(os.path.join(args.output, figname + ".png"))


def signatures_val(
    args,
    sign=["Ebf", "Eff", "Erc", "Epf"],
    start_time="2016-08-01 00:00",
    end_time="2018-08-01 00:00",
    figname="signatures_val",
    figsize=(12, 7),
):
    print("</> Plotting relative error of signatures...")

    # load model to validate
    model = smash.Model(
        *load_data(
            os.path.join(args.data, "info_bv.csv"),
            start_time=start_time,
            end_time=end_time,
            desc_dir="...",
        )
    )

    params = model.get_bound_constraints(states=False)["names"]

    df_sign = pd.DataFrame(columns=["code", "mapping"] + sign)

    for model_ddt, method in zip(args.models_ddt, args.methods):
        for par in params:
            setattr(model.parameters, par, model_ddt[par])

        model.run(inplace=True)

        res_sign = model.signatures(sign=sign, event_seg=dict(peak_quant=0.995))

        arr_obs = res_sign.event["obs"][sign].to_numpy()
        arr_sim = res_sign.event["sim"][sign].to_numpy()

        re = np.abs(arr_sim / arr_obs - 1)

        df = pd.DataFrame(data=re, columns=sign)

        df.insert(loc=0, column="code", value=res_sign.event["obs"]["code"].to_list())
        df.insert(loc=1, column="mapping", value=method)

        df_sign = pd.concat([df_sign, df], ignore_index=True)

    df_sign_1 = df_sign[df_sign["code"].isin(args.cal_code)]
    df_sign_1.insert(loc=2, column="type_val", value="Temp Val")

    df_sign_2 = df_sign[df_sign["code"].isin(args.val_code)]
    df_sign_2.insert(loc=2, column="type_val", value="Spatio-Temp Val")

    df_sign = pd.concat([df_sign_1, df_sign_2], ignore_index=True)

    # Create the plot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    for i in range(2):
        for j in range(2):
            st = sign[2 * i + j]

            sns.boxplot(
                data=df_sign,
                x="type_val",
                y=st,
                hue="mapping",
                width=0.5,
                palette="deep",
                showfliers=True,
                ax=axes[i, j],
                order=["Temp Val", "Spatio-Temp Val"],
            )

            # Set title and axis labels
            axes[i, j].set(title=None, xlabel=None, ylabel=f"RE of {st}")

            if i == 0:
                axes[i, j].set_xticklabels([])

            if j == 1:
                axes[i, j].set_yticklabels([])

            # Set y-axis limits
            axes[i, j].set_ylim([0, 1.2])

            axes[i, j].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])

            handles, labels = axes[
                i, j
            ].get_legend_handles_labels()  # get labels then remove
            axes[i, j].legend([], [], frameon=False)

    # Add legend
    fig.legend(
        handles,
        labels,
        title=None,
        loc="upper center",
        ncol=len(args.methods),
        fontsize=13,
    )

    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=0.15, hspace=0.1)

    plt.savefig(os.path.join(args.output, figname + ".png"))


def compare_signature_hist(
    args,
    sign=["Ebf", "Eff", "Erc", "Epf"],
    start_time="2016-08-01 00:00",
    end_time="2018-08-01 00:00",
    figname="compare_signature_hist",
    figsize=(15, 8),
):
    print("</> Plotting histograms...")

    colors = [
        "#4c72b0",
        "#dd8452",
        "#55a868",
        "#c44e52",
    ]  # corresponding colors for palette='deep

    # load model to validate
    model = smash.Model(
        *load_data(
            os.path.join(args.data, "info_bv.csv"),
            start_time=start_time,
            end_time=end_time,
            desc_dir="...",
        )
    )

    params = model.get_bound_constraints(states=False)["names"]

    df_sign = pd.DataFrame(columns=["code", "mapping"] + sign)

    for model_ddt, method in zip(args.models_ddt, args.methods):
        for par in params:
            setattr(model.parameters, par, model_ddt[par])

        model.run(inplace=True)

        res_sign = model.signatures(sign=sign, event_seg=dict(peak_quant=0.995))

        arr_obs = res_sign.event["obs"][sign].to_numpy()
        arr_sim = res_sign.event["sim"][sign].to_numpy()

        re = np.abs(arr_sim / arr_obs - 1)

        df = pd.DataFrame(data=re, columns=sign)

        df.insert(loc=0, column="code", value=res_sign.event["obs"]["code"].to_list())
        df.insert(loc=1, column="mapping", value=method)

        df_sign = pd.concat([df_sign, df], ignore_index=True)

    df_sign = df_sign[df_sign["code"].isin(args.cal_code)]

    # Create the plot
    fig, axes = plt.subplots(nrows=len(args.methods), ncols=len(sign), figsize=figsize)

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            x = df_sign[df_sign.mapping == args.methods[i]][sign[j]]
            x = x[np.isfinite(x)]

            axes[i, j].hist(x, bins=30, color=colors[i], range=[0, 2], density=True)

            # density = gaussian_kde(x)
            # x_vals = np.linspace(x.min(), x.max(), 100)
            # axes[i, j].plot(x_vals, density(x_vals), "black", linewidth=0.8)

            axes[i, j].set_xlim([0, 1.2])
            axes[i, j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
            axes[i, j].set_ylim([0, 4.5])
            axes[i, j].axvline(np.median(x), linewidth=1.5, label="Median", c="black")
            axes[i, j].axvline(
                np.mean(x), linewidth=1.5, label="Mean", c="black", linestyle="--"
            )
            axes[i, j].set_yticklabels([])
            axes[i, j].set_title(
                f"med={round(np.median(x), 2)}, mean={round(np.mean(x), 2)}, std={round(np.std(x), 2)}",
                fontsize=10,
            )

            if i < axes.shape[0] - 1:
                axes[i, j].set_xticklabels([])

            else:
                axes[i, j].tick_params(axis="x", labelsize=10)
                axes[i, j].set_xlabel(sign[j], fontsize=13)

            if j == 0:
                axes[i, j].set_ylabel(args.methods[i], fontsize=12)

    # Add a legend to the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12)

    plt.savefig(os.path.join(args.output, figname + ".png"))


##########
## MAIN ##
##########

if __name__ == "__main__":
    # Increase font size
    sns.set(font_scale=1.2)

    args = initialize_args()

    # cost_descent(args, niter=262)

    hydrograph(args, "cal", "hydrograph_cal")
    hydrograph(args, "val", "hydrograph_val")

    radialplot(args)
    boxplot_and_scatterplot(args)

    compare_signature_hist(args)

    param_map(args)

    desc_map(args)

    linear_cov(args)

    signatures_val(args)
