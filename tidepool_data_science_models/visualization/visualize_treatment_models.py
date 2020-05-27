"""
Visual explanations of the models.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("seaborn-poster")  # sets the size of the charts
style.use("ggplot")

from tidepool_data_science_models.models.treatment_models import PalermInsulinModel, CesconCarbModel
from tidepool_data_science_models.utils import get_timeseries, get_figure_filename

figures_path = os.path.join(os.getcwd(), "../../reports/figures")


# ------ Plot insulin models -----------
def plot_insulin_models(isf, cir, num_hours, insulin_amount, save_plot=False):

    insulin_models_to_plot = [PalermInsulinModel]

    plt.figure()

    for insulin_model in insulin_models_to_plot:
        model = insulin_model(isf=isf, cir=cir)
        t, delta_bg, bg, iob = model.run(
            num_hours=num_hours, insulin_amount=insulin_amount, five_min=True
        )
        plt.plot(t, bg, label=model.get_name())

    plt.ylabel("Blood Glucose (mg/dL)")
    plt.xlabel("Time (min)")
    plt.title(
        "Example Blood Glucose Response for Supported Insulin Models\nInsulin Amount={} U, ISF={} mg/dL / U".format(
            insulin_amount, isf
        )
    )
    plt.legend()

    if save_plot:
        figure_name = get_figure_filename(
            short_name="Insulin_Models_Plot",
            dataset_name="Simulated",
            version="v0.1",
            extension="png",
        )
        plt.savefig(os.path.join(figures_path, figure_name))


# ----- Plot carb models -------
def plot_carb_models(isf, cir, num_hours, carb_amount, save_plot=False):

    carb_models_to_plot = [CesconCarbModel]

    plt.figure()

    for carb_model in carb_models_to_plot:
        model = carb_model(isf=isf, cir=cir)
        t, delta_bg, bg = model.run(
            num_hours=num_hours, carb_amount=carb_amount, five_min=True
        )
        plt.plot(t, bg, label=model.get_name())

    plt.ylabel("Blood Glucose (mg/dL)")
    plt.xlabel("Time (min)")
    plt.title(
        "Example Blood Glucose Response for Supported Carb Models\nCarb Amount={} g, CIR={} g/U".format(
            carb_amount, cir
        )
    )
    plt.legend()
    if save_plot:
        figure_name = get_figure_filename(
            short_name="Carb_Models_Plot",
            dataset_name="Simulated",
            version="v0.1",
            extension="png",
        )
        plt.savefig(os.path.join(figures_path, figure_name))

    plt.show()


if __name__ == "__main__":

    insulin_amount = 1.0
    carb_amount = 10.0
    isf = 100
    cir = 10
    num_hours = 8

    # Make this true to save a new plot
    save_plot = True

    plot_insulin_models(
        isf=isf,
        cir=cir,
        num_hours=num_hours,
        insulin_amount=insulin_amount,
        save_plot=save_plot,
    )

    plot_carb_models(
        isf=isf,
        cir=cir,
        num_hours=num_hours,
        carb_amount=carb_amount,
        save_plot=save_plot,
    )
