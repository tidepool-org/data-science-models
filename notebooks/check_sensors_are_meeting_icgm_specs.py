import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
from tidepool_data_science_models.models.icgm_sensor_generator_functions import create_dataset, generate_icgm_sensors
import plotly.express as px

df = pd.read_csv(os.path.abspath(
    os.path.join("..", "data", "train_0a1f3ac86f7620ee531a6131bdc7844f57b6f70a3a7d451f0b0e40a8dff14dc9.csv_condition1.csv"
)))

true_bg_trace = df.iloc[50, 2:].astype(float).values
px.scatter(x=range(0, 2880), y=true_bg_trace).show()

# let's check the percent passing
batch_training_size = 30
sensor_generator = iCGMSensorGenerator(
    batch_training_size=batch_training_size,
    max_number_of_spurious_events_per_10_days=0,
    # use_g6_accuracy_in_loss=True,
    verbose=True
)

sensor_generator.fit(true_bg_trace)
# write this to file
sensor_generator.percent_pass
sensor_generator.icgm_special_controls_accuracy_table
sensor_generator.g6_table


# I would have two outputs
# one dataframe of the file and percent pass

# I would also just save the icgm_special_controls_accuracy_table
asdf = pd.DataFrame(sensor_generator.icgm_special_controls_accuracy_table)
asdf.to_csv("asdf.csv")