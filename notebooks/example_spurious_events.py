import datetime
import numpy as np
import matplotlib.pyplot as plt
from tidepool_data_science_models.models.icgm_sensor import iCGMSensor
from tidepool_data_science_models.models.icgm_sensor_generator_functions import create_dataset

test_sensor_properties = {
    "bias_type": "percentage_of_value",
    "a": 0,
    "b": 1,
    "mu": 0,
    "sigma": 1,
    "bias_drift_type": "none",
    "noise_coefficient": 0,
    "delay": 10,
    "max_number_of_spurious_events_per_sensor_life": 20,
}

sensor_datetime = datetime.datetime(2020, 1, 1)
icgm_sensor = iCGMSensor(current_datetime=sensor_datetime, sensor_properties=test_sensor_properties)

# make sure that the number of spurious events is less than or equal to the max allowable
assert (
    icgm_sensor.sensor_properties["spurious"].sum()
    <= test_sensor_properties["max_number_of_spurious_events_per_sensor_life"]
)

# generate a flat line and count the number of spurious events
true_df, _ = create_dataset(kind="flat", N=288 * 10, time_interval=5, flat_value=140, random_seed=0,)
true_bg_trace = true_df["value"].values
for expected_time_index, true_bg_val in enumerate(true_bg_trace):
    icgm_sensor.update(sensor_datetime, patient_true_bg=true_bg_val)
    sensor_datetime += datetime.timedelta(minutes=5)

icgm_true_diff = icgm_sensor.sensor_bg_history[2:] - true_bg_trace[2:]

n_spurious_sensed = np.sum(np.log(abs(np.mean(icgm_true_diff) - icgm_true_diff)) > 1)
assert n_spurious_sensed == icgm_sensor.sensor_properties["spurious_events_per_sensor"]

plt.plot(true_bg_trace)
plt.plot(icgm_sensor.sensor_bg_history)
plt.title("{} Spurious Events".format(n_spurious_sensed))
plt.show()
