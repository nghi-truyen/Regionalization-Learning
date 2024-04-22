import os
import argparse
import pickle
import random

import seaborn as sns
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib as mpl

import smash

if smash.__version__ >= "1.0":
    print("===================================")
    print(f"smash version: {smash.__version__}")
    print("===================================")

else:
    raise ValueError(
        "This code requires a minimum version of smash 1.0 or higher. Please update your smash installation."
    )


##################################
##  PARSER AND COMMON VARIABLES ##
##################################


def initialize_args():  # do not set new attr or modify any attr of args outside this function
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "-modeldir", "--modeldir", type=str, help="Select the model directory"
    )

    parser.add_argument(
        "-g",
        "-gauge",
        "--gauge",
        type=str,
        help="Select gauge type that has been used in the calibration process",
        choices=["upstream", "downstream", "intermediate", "independent"],
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

    args.methods = [
        "Uniform",
        "Multi-linear",
        "ANN",
    ]

    return args


##############################
##  FUNCTIONS FOR ANALYSIS  ##
##############################


def boxplot_scores(args, fobj="NSE", figname="scores", figsize=(15, 8)):
    print("</> Plotting boxplots...")

    df_reg_p1 = pd.read_csv(os.path.join(args.modeldir, "p1/scores.csv"))
    df_loc_p1 = pd.read_csv(
        os.path.join(os.path.dirname(args.modeldir), "local/p1/scores.csv")
    )
    df_loc_p1 = df_loc_p1.drop(columns=["nature"])
    df_p1 = pd.merge(df_reg_p1, df_loc_p1, on="code", how="right")

    df_reg_p2 = pd.read_csv(os.path.join(args.modeldir, "p2/scores.csv"))
    df_loc_p2 = pd.read_csv(
        os.path.join(os.path.dirname(args.modeldir), "local/p2/scores.csv")
    )
    df_loc_p2 = df_loc_p2.drop(columns=["nature"])
    df_p2 = pd.merge(df_reg_p2, df_loc_p2, on="code", how="right")

    df_merge = []

    n_gauged = len(df_p1[df_p1["nature"] == args.gauge])
    n_ungauged = len(df_p1) - n_gauged

    labels_col = [
        f"Cal ({n_gauged})",
        f"S_Val ({n_ungauged})",
        f"T_Val ({n_gauged})",
        f"S-T_Val ({n_ungauged})",
    ]

    for i, dfreg in enumerate([df_p1, df_p2]):
        # Melt the DataFrame to convert it to long format
        df = pd.melt(
            dfreg[
                ["code", "nature", "domain"]
                + [f"{fobj}_Uniform_loc", f"{fobj}_Distributed_loc"]
                + [f"{fobj}_{m}" for m in args.methods]
            ],
            id_vars=["code", "nature", "domain"],
            var_name="Metric",
            value_name=fobj,
        )

        # Extract the mapping information from the Metric column
        df["Mapping"] = df["Metric"].apply(
            lambda x: x.split("_")[1] + " (reg)"
            if len(x.split("_")) < 3
            else x.split("_")[1] + " (loc)"
        )
        mapping_order = df["Mapping"].unique()

        # Drop the Metric column
        df = df.drop(columns=["Metric"])

        # Replace "cal" and "val" in the 'domain' column
        df["domain"] = df["domain"].replace(
            {
                "cal": labels_col[0] if i == 0 else labels_col[2],
                "val": labels_col[1] if i == 0 else labels_col[3],
            }
        )

        df_merge.append(df)

    df_merge = pd.concat(df_merge, ignore_index=True)

    colors = [
        "#a2cffe",
        "#fed0fc",
        "#4c72b0",
        "#dd8452",
        "#55a868",
    ]

    # Create the plot
    plt.figure(figsize=figsize)

    # Boxplot
    bplt = sns.boxplot(
        data=df_merge,
        x="domain",
        y=fobj,
        hue="Mapping",
        hue_order=mapping_order,
        width=0.5,
        palette=colors,
        showfliers=False,
        order=labels_col,
    )
    # Set hatch pattern
    hatch = "/////"
    mpl.rcParams["hatch.linewidth"] = 0.6
    for i in range(4):
        bplt.patches[i].set_hatch(hatch)
    for i in range(2, 5):
        bplt.patches[i * len(colors)].set_hatch(hatch)
        bplt.patches[i * len(colors) + 1].set_hatch(hatch)

    # Set title and axis labels
    plt.title(f"Cal {args.gauge}", loc="left")
    plt.xlabel(None)
    plt.ylabel(fobj)

    # Set y-axis limits
    plt.ylim([-0.25, 1])

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()  # get labels then remove
    plt.legend(
        handles,
        labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.61, 1.18),
        ncol=4,
        fontsize=14,
    )

    plt.savefig(os.path.join(args.output, figname + ".png"))


def boxplot_scores_by_nature(
    args, fobj="NSE", figname="scores_by_nature", figsize=(15, 8)
):
    print("</> Plotting boxplots...")

    df_reg_p1 = pd.read_csv(os.path.join(args.modeldir, "p1/scores.csv"))
    df_reg_p2 = pd.read_csv(os.path.join(args.modeldir, "p2/scores.csv"))

    n_count = df_reg_p1["nature"].value_counts()

    df_merge = []
    col_labels = []

    for t_val, dfreg in zip(["P1, S_Val", "P2, S-T_Val"], [df_reg_p1, df_reg_p2]):
        dfreg = dfreg[dfreg["domain"] == "val"]
        dfreg = dfreg.drop(columns=["domain"])
        # Melt the DataFrame to convert it to long format
        df = pd.melt(
            dfreg[["code", "nature"] + [f"{fobj}_{m}" for m in args.methods]],
            id_vars=["code", "nature"],
            var_name="Metric",
            value_name=fobj,
        )

        # Extract the mapping information from the Metric column
        df["Mapping"] = df["Metric"].apply(lambda x: x.split("_")[1])

        # Drop the Metric column
        df = df.drop(columns=["Metric"])

        du = "upstream" if args.gauge == "downstream" else "downstream"
        dict_count = {
            nature: f"{nature.capitalize()} ({n_count[nature]})\n{t_val}"
            for nature in [du, "intermediate", "independent"]
        }
        df["nature"] = df["nature"].replace(dict_count)

        df_merge.append(df)
        col_labels += list(dict_count.values())

    df_merge = pd.concat(df_merge, ignore_index=True)

    # Create the plot
    plt.figure(figsize=figsize)

    # Boxplot
    sns.boxplot(
        data=df_merge,
        x="nature",
        y=fobj,
        hue="Mapping",
        hue_order=args.methods,
        width=0.5,
        palette="deep",
        showfliers=False,
        order=col_labels,
    )

    # Set title and axis labels
    plt.title(f"Cal {args.gauge}", loc="left")
    plt.xlabel(None)
    plt.ylabel(fobj)

    # Set y-axis limits
    plt.ylim([-0.25, 1])

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()  # get labels then remove
    plt.legend(
        handles,
        labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(args.methods),
        fontsize=14,
    )

    plt.savefig(os.path.join(args.output, figname + ".png"))


def hydrograph(
    args,
    period="p1",
    figname="hydrograph_p1",
    extration=None,
    figsize: tuple | None = None,
):  # figsize=None: auto adjusted figure size
    print(f"</> Plotting hydrograph ({period})...")

    dti = pd.date_range(
        start="2016-08-01" if period == "p1" else "2020-08-01",
        end="2020-07-31" if period == "p1" else "2022-07-31",
        freq="h",
    )[1:]

    df = pd.read_csv(os.path.join(args.modeldir, f"{period}/scores.csv"))

    codes = df["code"].to_list()
    indices = df.index.tolist()

    if not extration is None:  # randomly extracting hydrograph to be plotted
        indices, codes = zip(*random.sample(list(zip(indices, codes)), extration))

    nr = len(codes)
    nc = len(args.methods)

    if figsize == None:
        figsize = (3 * nc, nr)

    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=figsize)

    for j, mtd in enumerate(args.methods):
        with open(
            os.path.join(args.modeldir, f"{period}/{mtd}_discharges.pickle"), "rb"
        ) as f:
            q = pickle.load(f)

        axes[0, j].set_title(mtd, fontsize=12)

        for i, ind in enumerate(indices):
            qo = np.copy(q["obs"][ind, :])
            qs = np.copy(q["sim"][ind, :])

            qo[np.where(qo < 0)] = np.nan
            qs[np.where(qs < 0)] = np.nan

            ax = axes[i, j]

            ax.plot(dti, qo, color="red", label="Observed", linewidth=2)
            ax.plot(
                dti, qs, color="blue", linestyle="-.", label="Simulated", linewidth=1
            )

            ax.tick_params(axis="both", which="both", labelsize=10)

            if i < len(indices) - 1:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis="x", labelrotation=60)

            ax.xaxis.grid(False)

            if j > 0:  # remove ticklabel
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(
                    codes[i][:-2] + f"\n({df['nature'].iloc[ind]})", fontsize=10
                )

    # set the same scale for y-axis
    for i in range(axes.shape[0]):
        ymin = min([ax.get_ylim()[0] for ax in axes[i, :]])

        ymax = max([ax.get_ylim()[1] for ax in axes[i, :]])

        for j in range(axes.shape[1]):
            axes[i, j].set_ylim([ymin, ymax])

    # Add a legend to the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)

    plt.savefig(os.path.join(args.output, figname + ".png"))


def param_map(
    args,
    params=["cp", "ct", "kexc", "llr"],
    math_params=[r"$c_p$", r"$c_t$", r"$k_{exc}$", r"$l_{l_r}$"],
    bounds=[(0, 1000), (0, 1000), (-50, 50), (0, 1000)],
    cmap="Spectral",
    figname="param_map",
    figsize=(10, 7),
):
    print("</> Plotting parameters map...")

    fig, axes = plt.subplots(
        nrows=len(args.methods[1:]), ncols=len(params), figsize=figsize
    )

    for i, method in enumerate(args.methods[1:]):
        with open(
            os.path.join(args.modeldir, f"p1/{method}_parameters.pickle"), "rb"
        ) as f:
            parameters = pickle.load(f)

        for j, par in enumerate(params):
            if i == 0:
                axes[0, j].set_title(math_params[j], fontsize=14)

            axes[i, j].yaxis.grid(False)
            axes[i, j].xaxis.grid(False)

            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            value = parameters[par]
            mask = parameters["mask_ac"]

            im = axes[i, j].imshow(
                value,
                cmap=cmap,
                vmin=bounds[j][0],
                vmax=bounds[j][1],
                interpolation="bicubic",
                alpha=1.0,
            )

            mu = np.mean(value[mask])
            std = np.std(value[mask])
            xlabel = f"$\mu$={str(round(mu, 1))}, $\sigma$={str(round(std, 1))}"

            axes[i, j].set_xlabel(xlabel, labelpad=5, fontsize=12)

            if j == 0:
                axes[i, j].set_ylabel(
                    f"Cal {args.gauge}\n{method}", labelpad=10, fontsize=12
                )

            # Add a colorbar to the new axes
            clb = fig.colorbar(im, orientation="horizontal")
            # Set fontsize for colorbar
            clb.ax.tick_params(labelsize=10)

    plt.savefig(os.path.join(args.output, figname + ".png"))


def desc_map(
    cmap="terrain",
    figname="desc_map",
    figsize=(15, 4),
):
    print("</> Plotting descriptors map...")

    with open("data/descriptors.pickle", "rb") as f:
        descriptor = pickle.load(f)

    fig, axes = plt.subplots(nrows=1, ncols=len(descriptor.keys()), figsize=figsize)

    for i, darr in enumerate(descriptor.values()):
        axes[i].set_title(rf"$d_{i + 1}$", fontsize=12)

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
    params=["cp", "ct", "kexc", "llr"],
    math_params=[r"$c_p$", r"$c_t$", r"$k_{exc}$", r"$l_{l_r}$"],
    figname="linear_cov",
    figsize=(6, 4),
):
    print("</> Plotting linear covariance matrix...")

    with open("data/descriptors.pickle", "rb") as f:
        descriptor = pickle.load(f)

    fig, axes = plt.subplots(
        nrows=len(args.methods[1:]), ncols=1, figsize=figsize, constrained_layout=True
    )

    for k, method in enumerate(args.methods[1:]):
        with open(
            os.path.join(args.modeldir, f"p1/{method}_parameters.pickle"), "rb"
        ) as f:
            parameters = pickle.load(f)

        cov_mat = np.zeros((len(params), len(descriptor.keys())))

        for j, dei in enumerate(descriptor.values()):
            for i, par_name in enumerate(params):
                pai = parameters[par_name]

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

        ytl = [rf"$d_{num + 1}$" for num in range(len(descriptor.keys())) if k == 0]

        axes[k].yaxis.grid(False)
        axes[k].xaxis.grid(False)

        sns.heatmap(
            cov_mat,
            xticklabels=ytl,
            yticklabels=math_params,
            vmin=0,
            vmax=1,
            square=True,
            cbar=(k == len(args.methods[1:]) - 1),
            cbar_kws=dict(
                use_gridspec=False, location="bottom", shrink=0.55, aspect=40
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

        axes[k].set_ylabel(
            f"Cal {args.gauge}\n{args.methods[k + 1]}", fontsize=12, labelpad=10
        )

    plt.savefig(os.path.join(args.output, figname + ".png"))


def boxplot_signatures(
    args,
    sign=["Eff", "Ebf", "Erc", "Elt", "Epf"],
    figname="signatures",
    figsize=(12, 7),
):
    print("</> Plotting relative error of signatures...")

    dfp1 = []
    dfp2 = []
    for mtd in args.methods:
        with open(
            os.path.join(args.modeldir, f"p1/{mtd}_signatures.pickle"), "rb"
        ) as f:
            signatures = pickle.load(f)
            df_obs = signatures["obs"].event[sign]
            df_sim = signatures["sim"].event[sign]

            df = (df_sim - df_obs) / df_obs
            df = df.abs()
            df = df.add_prefix(mtd + "_")
            codes_p1 = signatures["obs"].event["code"].to_list()
        dfp1.append(df)

        with open(
            os.path.join(args.modeldir, f"p2/{mtd}_signatures.pickle"), "rb"
        ) as f:
            signatures = pickle.load(f)
            df_obs = signatures["obs"].event[sign]
            df_sim = signatures["sim"].event[sign]

            df = (df_sim - df_obs) / df_obs
            df = df.abs()
            df = df.add_prefix(mtd + "_")
            codes_p2 = signatures["obs"].event["code"].to_list()
        dfp2.append(df)

    df_reg = pd.read_csv(os.path.join(args.modeldir, "p1/scores.csv"))

    dfp1 = pd.concat(dfp1, axis=1)
    dfp1.insert(0, "code", codes_p1)
    dfp1 = pd.merge(dfp1, df_reg[["code", "nature", "domain"]], on="code", how="left")

    dfp2 = pd.concat(dfp2, axis=1)
    dfp2.insert(0, "code", codes_p2)
    dfp2 = pd.merge(dfp2, df_reg[["code", "nature", "domain"]], on="code", how="left")

    df_merge = []

    n_gauged_p1 = len(
        dfp1[
            dfp1["code"].isin(df_reg[df_reg["nature"] == args.gauge]["code"]).to_list()
        ]
    )
    n_ungauged_p1 = len(dfp1) - n_gauged_p1
    n_gauged_p2 = len(
        dfp2[
            dfp2["code"].isin(df_reg[df_reg["nature"] == args.gauge]["code"]).to_list()
        ]
    )
    n_ungauged_p2 = len(dfp2) - n_gauged_p2

    labels_col = [
        f"Cal ({n_gauged_p1})",
        f"S_Val ({n_ungauged_p1})",
        f"T_Val ({n_gauged_p2})",
        f"S-T_Val ({n_ungauged_p2})",
    ]

    # Create subplots
    fig, axes = plt.subplots(1, len(sign), figsize=figsize)
    axes[0].set_title(f"Cal {args.gauge}", loc="left")

    for idx, fobj in enumerate(sign):
        # Reset df_merge for each fobj
        df_merge = []

        for i, dfsign in enumerate([dfp1, dfp2]):
            # Melt the DataFrame to convert it to long format
            df = pd.melt(
                dfsign[
                    ["code", "nature", "domain"] + [f"{m}_{fobj}" for m in args.methods]
                ],
                id_vars=["code", "nature", "domain"],
                var_name="Metric",
                value_name=f"Relative Error ({fobj})",
            )

            # Extract the mapping information from the Metric column
            df["Mapping"] = df["Metric"].apply(lambda x: x.split("_")[0])

            # Drop the Metric column
            df = df.drop(columns=["Metric"])

            # Replace "cal" and "val" in the 'domain' column
            df["domain"] = df["domain"].replace(
                {
                    "cal": labels_col[0] if i == 0 else labels_col[2],
                    "val": labels_col[1] if i == 0 else labels_col[3],
                }
            )

            df_merge.append(df)

        df_merge = pd.concat(df_merge, ignore_index=True)

        # Plot on the current subplot
        sns.boxplot(
            data=df_merge,
            x="domain",
            y=f"Relative Error ({fobj})",
            hue="Mapping",
            hue_order=args.methods,
            width=0.5,
            palette="deep",
            showfliers=False,
            order=labels_col,
            ax=axes[idx],  # Set the current subplot
        )

        axes[idx].set_xlabel(None)
        axes[idx].set_ylabel(f"Relative Error ({fobj})", fontsize=12)
        # Set y-axis limits
        axes[idx].set_ylim([-0.01, 1.4])
        axes[idx].legend().remove()
        # Rotate x-axis labels
        axes[idx].set_xticklabels(labels_col, rotation=15, fontsize=12)
        # Set y-axis label fontsize
        axes[idx].tick_params(axis="y", labelsize=12)

    # Get handles and labels for legend from the last subplot
    handles, labels = axes[-1].get_legend_handles_labels()

    # Set legend for the entire figure
    plt.legend(
        handles,
        labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(-0.75, 1.15),
        ncol=len(args.methods),
        fontsize=14,
    )

    plt.savefig(os.path.join(args.output, figname + ".png"))


def cost_gradient(
    args, maxiter=360, figsize=(15, 9), figname="cost_projected_gradient"
):
    print("</> Plotting cost and projected gradient...")

    # Define colors and line styles for each method
    colors = [
        "#dd8452",
        "#55a868",
    ]  # corresponding colors for palette='deep'

    fig, axes = plt.subplots(nrows=len(args.methods[1:]), figsize=figsize)

    for i, mtd in enumerate(args.methods[1:]):
        with open(os.path.join(args.modeldir, f"p1/{mtd}_ret.pickle"), "rb") as f:
            load_pickle = pickle.load(f)

            cost = load_pickle.iter_cost

            projg = load_pickle.iter_projg
            projg = np.array(projg)[:maxiter]

        axes[0].plot(
            cost,
            color=colors[i],
            label=mtd,
        )
        axes[1].plot(
            projg,
            color=colors[i],
            label=mtd,
        )

    for j in range(2):
        # Customize tick labels and grid lines
        axes[j].tick_params(axis="both", which="major", labelsize=12, width=2, length=6)
        axes[j].tick_params(axis="both", which="minor", labelsize=10, width=1, length=4)
        axes[j].grid(
            True,
            which="major",
            axis="both",
            linestyle="--",
            color="lightgray",
            alpha=0.7,
        )

        # Set x and y axis labels, title and legend
        axes[j].set_xlabel("Iteration/Epoch", fontsize=14)

        axes[j].set_xlim([-2, maxiter])

    axes[0].set_ylabel("Cost", fontsize=14)
    axes[1].set_ylabel("Proj_G", fontsize=14)

    plt.savefig(os.path.join(args.output, figname + ".png"))


##########
## MAIN ##
##########

if __name__ == "__main__":
    # Increase font size
    sns.set(font_scale=1.2)

    args = initialize_args()

    hydrograph(args, "p1", "hydrograph_p1", 4, figsize=(10, 6.5))
    hydrograph(args, "p2", "hydrograph_p2", 4, figsize=(10, 6.5))

    boxplot_scores(args, fobj="NSE", figsize=(15, 5.5))
    boxplot_scores(args, fobj="KGE", figsize=(15, 5.5))

    boxplot_signatures(args, figsize=(18.5, 5), sign=["Erc", "Eff", "Epf"])

    param_map(args, bounds=((150, 900), (0, 150), (-15, 5), (0, 150)))

    desc_map()

    linear_cov(args, figsize=(5, 4))

    boxplot_scores_by_nature(args, fobj="NSE", figsize=(15, 5))
    boxplot_scores_by_nature(args, fobj="KGE", figsize=(15, 5))

    cost_gradient(args)
