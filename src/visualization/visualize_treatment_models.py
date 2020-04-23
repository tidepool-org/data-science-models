"""
Visual explanations of the models.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("seaborn-poster")  # sets the size of the charts
style.use("ggplot")

from src.models.treatment_models import PalermInsulinModel, CesconCarbModel

figures_path = os.path.join(os.getcwd(), "../../reports/figures")

insulin_models_to_plot = [PalermInsulinModel]
carb_models_to_plot = [CesconCarbModel]

# ------ Plot insulin models -----------
isf = 100
cir = 10
insulin_amount = 1.0
num_hours = 8
for insulin_model in insulin_models_to_plot:
    model = insulin_model(isf=isf, cir=cir)
    t, delta_bg, bg, iob = model.run(num_hours=num_hours, insulin_amount=insulin_amount, five_min=True)
    plt.plot(t, bg, label=model.get_name())

plt.ylabel("Blood Glucose (mg/dL)")
plt.xlabel("Time (min)")
plt.title(
    "Example Blood Glucose Response for Supported Insulin Models\nInsulin Amount={} U, ISF={} mg/dL / U".format(
        insulin_amount, isf
    )
)
plt.legend()
# plt.savefig(os.path.join(figures_path, "insulin_models_plot.png"))


# ----- Plot carb models -------
plt.figure()

carb_amount = 10.0
for carb_model in carb_models_to_plot:
    model = carb_model(isf=isf, cir=cir)
    t, delta_bg, bg = model.run(num_hours=num_hours, carb_amount=carb_amount, five_min=True)
    plt.plot(t, bg, label=model.get_name())

plt.ylabel("Blood Glucose (mg/dL)")
plt.xlabel("Time (min)")
plt.title(
    "Example Blood Glucose Response for Supported Carb Models\nCarb Amount={} g, CIR={} g/U".format(
        carb_amount, cir
    )
)
plt.legend()
# plt.savefig(os.path.join(figures_path, "carb_models_plot.png"))

plt.show()