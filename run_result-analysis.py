import smash
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import seaborn as sns
from smash.solver._mwd_cost import nse as cost_nse
from mpl_toolkits.axes_grid1 import make_axes_locatable


print("===================================")
print(f"smash version: {smash.__version__}")
print("===================================")


METHODS = ["uniform", "distributed", "hyper-linear", "hyper-polynomial", "ann"]


parser = argparse.ArgumentParser()

parser.add_argument("-d", "-data", "--data", type=str, help="Select the data directory")

parser.add_argument(
    "-m", "-model", "--model", type=str, help="Select the model directory"
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

cal_code = pd.read_csv(os.path.join(args.data, "cal_code.csv"))["cal"].to_list()

val_code = pd.read_csv(os.path.join(args.data, "val_code.csv"))["val"].to_list()

###########
## BOXPLOTS
###########

print("</> Plotting boxplots...")

models = [
    smash.read_model_ddt(os.path.join(args.model, method + ".hdf5"))
    for method in METHODS
]


cost = []
metd = []

cal_val = []
code_catch = []

for i, code in enumerate(models[0]["code"]):

    for model, method in zip(models, METHODS):

        qo = model["qobs"][i]
        qs = model["qsim"][i]

        if code in cal_code:

            cal_val.append("Cal")

        elif code in val_code:

            cal_val.append("Val")

        else:
            continue

        cost.append(1 - cost_nse(qo, qs))

        metd.append(method)

        code_catch.append(code)

df = pd.DataFrame(
    {"code": code_catch, "cal_val": cal_val, "Mapping": metd, "NSE": cost}
)

# Increase font size
sns.set(font_scale=1.2)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
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
    handles[0 : len(METHODS)],
    labels[0 : len(METHODS)],
    title="Mapping Type",
    title_fontsize=14,
    loc="upper right",
    bbox_to_anchor=(1.3, 1),
)

# Set y-axis limits and add grid lines
ax.set_ylim([0, 1])
ax.yaxis.grid(True)

# Adjust plot spacing
plt.subplots_adjust(right=0.8)

plt.savefig(os.path.join(args.output, "boxplots.png"))


############
## PARAM MAP
############

print("</> Plotting parameters map...")

params = ["exc", "lr", "cft", "cp"]
bounds = [(-20, 5), (1, 200), (1, 1000), (2, 2000)]

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))

for j, par in enumerate(params):

    axes[0, j].set_title(params[j])

    for i, mod in enumerate(models[2:]):

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

            axes[i, j].set_ylabel(METHODS[i + 2], labelpad=10)

    divider = make_axes_locatable(axes[-1, j])
    cax = divider.new_vertical(size="5%", pad=0.2, pack_start=True)
    # Add a colorbar to the new axes
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation="horizontal")

plt.savefig(os.path.join(args.output, "param_map.png"))


# ###############
# ## COST DESCENT
# ###############

# print("</> Plotting cost descent...")

# x = range(1, 263)

# # Define colors and line styles for each method
# colors = ['#dd8452', '#55a868', '#c44e52', '#8172b3']  # corresponding colors in deep
# linestyles = ["--", ":", "-.", "-"]
# line_kwargs = {"linewidth": 2}

# fig, ax = plt.subplots(figsize=(12, 6))

# for i, mtd in enumerate(METHODS[1:]):
#     if mtd == "ann":
#         J = np.loadtxt(args.model + "outterminal/" + mtd + ".txt", usecols=(5, 9))
#     else:
#         J = np.loadtxt(args.model + "outterminal/" + mtd + ".txt", usecols=(8, 12))

#     ax.plot(x, J[: len(x), 0], label=mtd, color=colors[i], linestyle=linestyles[i], **line_kwargs)

# # Set x and y axis labels, title and legend
# ax.set_xlabel("Iteration", fontsize=14)
# ax.set_ylabel("Cost", fontsize=14)
# # ax.set_title("Multisites Optimization", fontsize=16, fontweight="bold")
# ax.legend(fontsize=12)

# # Customize tick labels and grid lines
# ax.tick_params(axis="both", which="major", labelsize=12, width=2, length=6)
# ax.tick_params(axis="both", which="minor", labelsize=10, width=1, length=4)
# ax.grid(True, which="major", axis="both", linestyle="--", color="lightgray", alpha=0.7)

# plt.savefig(os.path.join(args.output, "cost_descent.png"))
