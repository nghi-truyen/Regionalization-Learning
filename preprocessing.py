import smash
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data(
    catchment: str | pd.DataFrame,
    structure="gr-b",
    start_time="2006-08-01 00:00",
    end_time="2016-08-01 00:00",
    descriptor_name=[
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

    flowdir = smash.load_dataset("flwdir")

    if not isinstance(catchment.Code_BV, str):

        mesh = smash.generate_mesh(
            flowdir,
            x=catchment.Xexu.values,
            y=catchment.Yexu.values,
            area=catchment.Surf_Bnbv.values * 10**6,
            code=catchment.Code_BV.values,
        )

    else:
        mesh = smash.generate_mesh(
            flowdir,
            x=catchment.Xexu,
            y=catchment.Yexu,
            area=catchment.Surf_Bnbv * 10**6,
            code=catchment.Code_BV,
        )

    setup = {
        "structure": structure,
        "dt": 3600,
        "start_time": start_time,
        "end_time": end_time,
        "read_qobs": True,
        "qobs_directory": "../DATA/qobs",
        "read_prcp": True,
        "prcp_format": "tif",
        "prcp_conversion_factor": 0.1,
        "prcp_directory": "/home/RHAX/DONNEES/PLUIE/SPATIAL/ANTLP/L93-FRA/J+1/1H",
        "read_pet": True,
        "pet_format": "tif",
        "pet_conversion_factor": 1,
        "daily_interannual_pet": True,
        "pet_directory": "/home/RHAX/DONNEES/ETP/GRILLES/ETP-SFR-FRA-INTERA_L93",
        "read_descriptor": True,
        "descriptor_name": descriptor_name,
        "descriptor_directory": "../DATA/descriptor",
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

            des = model.input_data.descriptor[..., i]

            plt.imshow(des)

            plt.colorbar()

            plt.title(d)

            plt.show()
