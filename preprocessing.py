import smash
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data(
    catchment: str | pd.DataFrame,
    start_time: str | pd.Series,
    end_time: str | pd.Series,
    q_dir="../DATA/qobs",
    prcp_dir="../DATA/prcp",
    pet_dir="../DATA/pet",
    desc_dir="../DATA/descriptor",
    desc_name=[
        "pente",
        "ddr",
        "karst2019_shyreg",
        "foret",
        "urbain",
        "resutilpot",
        "vhcapa",
        "grassland",
        "medcapa",
        "arable",
    ],
):
    if isinstance(catchment, (pd.Series, pd.DataFrame)):
        pass

    elif isinstance(catchment, str):
        catchment = pd.read_csv(catchment)

    else:
        raise TypeError(
            f"catchment must be str or DataFrame, and not {type(catchment)}"
        )

    flowdir = smash.factory.load_dataset("flwdir")

    if not isinstance(catchment.code, str):
        mesh = smash.factory.generate_mesh(
            flowdir,
            x=catchment.x.values,
            y=catchment.y.values,
            area=catchment.area.values * 10**6,
            code=catchment.code.values,
        )

    else:
        mesh = smash.factory.generate_mesh(
            flowdir,
            x=catchment.x,
            y=catchment.y,
            area=catchment.area * 10**6,
            code=catchment.code,
        )

    setup = {
        "dt": 3600,
        "start_time": start_time,
        "end_time": end_time,
        "read_qobs": True,
        "qobs_directory": q_dir,
        "read_prcp": True,
        "prcp_format": "tif",
        "prcp_conversion_factor": 0.1,
        "prcp_directory": prcp_dir,
        "read_pet": True,
        "pet_format": "tif",
        "pet_conversion_factor": 1,
        "daily_interannual_pet": True,
        "pet_directory": pet_dir,
        "read_descriptor": False if desc_dir == "..." else True,
        "descriptor_name": desc_name,
        "descriptor_directory": desc_dir,
        "sparse_storage": True,
    }

    return setup, mesh


def preprocess_visualize(setup, mesh, descriptor_plot=True):
    plt.imshow(mesh["flwdst"])

    plt.colorbar(label="Flow distance (m)")

    plt.title("Nested multiple gauge - Flow distance")

    plt.show()

    canvas = np.zeros(shape=mesh["flwdir"].shape)

    canvas = np.where(mesh["active_cell"] == 0, np.nan, canvas)

    for pos in mesh["gauge_pos"]:
        canvas[tuple(pos)] = 1

    plt.imshow(canvas, cmap="Set1_r")

    plt.title("Nested multiple gauge - Gauges location")

    plt.show()

    if descriptor_plot:
        setup_copy = setup.copy()

        setup_copy["end_time"] = pd.Timestamp(setup_copy["start_time"]) + pd.Timedelta(
            days=1
        )

        model = smash.Model(setup_copy, mesh)

        for i, d in enumerate(setup_copy["descriptor_name"]):
            des = model.physio_data.descriptor[..., i]

            plt.imshow(des)

            plt.colorbar()

            plt.title(d)

            plt.show()
