"""
Visual explanations of the models
"""
import os
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("seaborn-poster")  # sets the size of the charts
style.use("ggplot")

from tidepool_data_science_models.models.treatment_models import PalermInsulinModel, CesconCarbModel
from tidepool_data_science_models.utils import get_timeseries

figures_path = os.path.join(os.getcwd(), "../../reports/figures")

insulin_models_to_plot = [PalermInsulinModel]
carb_models_to_plot = [CesconCarbModel]

# Plot insulin models
t = get_timeseries(8, five_min=False)
t_5min = get_timeseries(8, five_min=True)

plt.figure()
isf = 100
cir = 10
insulin_amount = 1.0
for insulin_model in insulin_models_to_plot:
    model = insulin_model(isf=isf, cir=cir)
    i_t, iob_t = model.run(t=t, insulin_amount=insulin_amount)
    i_5min, iob_5min = model.run(t=t_5min, insulin_amount=insulin_amount)
    plt.plot(t, i_t, label=model.get_name())
    # plt.plot(t_5min, i_5min)

plt.ylabel("Blood Glucose (mg/dL)")
plt.xlabel("Time (min)")
plt.title(
    "Example Blood Glucose Response for Supported Insulin Models\nInsulin Amount={} U, ISF={} mg/dL / U".format(
        insulin_amount, isf
    )
)
plt.legend()
plt.savefig(os.path.join(figures_path, "insulin_models_plot.png"))


# ----- Plot carb models -------
plt.figure()

carb_amount = 10.0
for carb_model in carb_models_to_plot:
    model = carb_model(isf=isf, cir=cir)
    c_t = model.run(t=t, carb_amount=carb_amount)
    plt.plot(t, c_t, label=model.get_name())

plt.ylabel("Blood Glucose (mg/dL)")
plt.xlabel("Time (min)")
plt.title(
    "Example Blood Glucose Response for Supported Carb Models\nCarb Amount={} g, CIR={} g/U".format(
        carb_amount, cir
    )
)
plt.legend()
plt.savefig(os.path.join(figures_path, "carb_models_plot.png"))
