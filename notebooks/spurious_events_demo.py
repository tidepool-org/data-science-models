import numpy as np
import matplotlib.pyplot as plt
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
from tidepool_data_science_models.models.icgm_sensor_generator_functions import (
    create_dataset,
    generate_icgm_sensors,
    preprocess_data,
    calc_icgm_sc_table,
)


true_df, _ = create_dataset(
    kind="sine",
    N=288 * 10,
    min_value=40,
    max_value=400,
    time_interval=5,
    flat_value=np.nan,
    oscillations=5,
    random_seed=0,
)

true_bg_trace = true_df["value"].values
plt.plot(true_bg_trace)
plt.show()

dist_params = [1, 1, 1, 1]

# %% first confirm that if number of spurious events is set to zero, that there are not spurious events
number_of_spurious_events_per_10_days = 0

icgm_trace, ind_sensor_properties = generate_icgm_sensors(
    true_bg_trace,
    dist_params,  # [a, b, mu, sigma]
    n_sensors=1,  # (suggest 100 for speed, 1000 for thoroughness)
    bias_type="percentage_of_value",  # (constant_offset, percentage_of_value)
    bias_drift_type="random",  # options (none, linear, random)
    bias_drift_range=[0.85, 1.15],  # (suggest keeping within +/-15%)
    bias_drift_oscillations=1,  # opt for random drift (max of 2)
    noise_coefficient=2.5,  # (0 ~ 60dB, 5 ~ 36 dB, 10, 30 dB)
    delay=10,  # (suggest 0, 5, 10, 15)
    number_of_spurious_events_per_10_days=number_of_spurious_events_per_10_days,
    random_seed=0,
)
plt.plot(icgm_trace.T)
plt.show()

# %% visually confirm the number of spurious events is correct and that the random seed puts the spurious event in different locations
true_df, _ = create_dataset(kind="flat", N=288 * 10, time_interval=5, flat_value=140, random_seed=0,)

true_bg_trace = true_df["value"].values
plt.plot(true_bg_trace)
plt.show()

for r in range(20):
    icgm_trace, ind_sensor_properties = generate_icgm_sensors(
        true_bg_trace,
        dist_params,  # [a, b, mu, sigma]
        n_sensors=1,  # (suggest 100 for speed, 1000 for thoroughness)
        bias_type="percentage_of_value",  # (constant_offset, percentage_of_value)
        bias_drift_type="none",  # options (none, linear, random)
        bias_drift_range=[0.85, 1.15],  # (suggest keeping within +/-15%)
        bias_drift_oscillations=1,  # opt for random drift (max of 2)
        noise_coefficient=0,  # (0 ~ 60dB, 5 ~ 36 dB, 10, 30 dB)
        delay=0,  # (suggest 0, 5, 10, 15)
        number_of_spurious_events_per_10_days=r,
        random_seed=r,
    )

    icgm_true_diff = icgm_trace - true_bg_trace
    n_spurious_sensed = np.sum(np.log(abs(np.mean(icgm_true_diff) - icgm_true_diff)) > 1)
    print(n_spurious_sensed)
    assert n_spurious_sensed == r

    plt.plot(icgm_trace.T)
    plt.show()


# %% test that the value of the spurious events is within icgm special controls
true_df, _ = create_dataset(kind="linear", N=288 * 10, min_value=40, max_value=400, time_interval=5, random_seed=0,)
true_bg_trace = true_df["value"].values
plt.plot(true_bg_trace.T)
plt.show()

icgm_trace, ind_sensor_properties = generate_icgm_sensors(
    true_bg_trace,
    dist_params,  # [a, b, mu, sigma]
    n_sensors=1,  # (suggest 100 for speed, 1000 for thoroughness)
    bias_type="percentage_of_value",  # (constant_offset, percentage_of_value)
    bias_drift_type="none",  # options (none, linear, random)
    bias_drift_range=[0.85, 1.15],  # (suggest keeping within +/-15%)
    bias_drift_oscillations=1,  # opt for random drift (max of 2)
    noise_coefficient=0,  # (0 ~ 60dB, 5 ~ 36 dB, 10, 30 dB)
    delay=0,  # (suggest 0, 5, 10, 15)
    number_of_spurious_events_per_10_days=1000,
    random_seed=0,
)

plt.plot(icgm_trace.T)
plt.show()

"""
(H) When iCGM values are less than 70 mg/dL, no corresponding blood glucose value shall read above 180 mg/dL.
(I) When iCGM values are greater than 180 mg/dL, no corresponding blood glucose value shall read less than 70 mg/dL.
"""

icgm_special_controls_table = calc_icgm_sc_table(
    preprocess_data(true_bg_trace, icgm_trace, icgm_range=[40, 400], ysi_range=[0, 900])
)

assert icgm_special_controls_table.loc["H", "icgmSensorResults"] == 100
assert icgm_special_controls_table.loc["I", "icgmSensorResults"] == 100


# %% test that the spurious events work in the icgm sensor generator
true_df, _ = create_dataset(
    kind="sine",
    N=288 * 10,
    min_value=40,
    max_value=400,
    time_interval=5,
    flat_value=np.nan,
    oscillations=5,
    random_seed=0,
)

true_bg_trace = true_df["value"].values
plt.plot(true_bg_trace)
plt.show()


# create a batch of sensors
batch_training_size = 30
sensor_generator = iCGMSensorGenerator(
    batch_training_size=batch_training_size, max_number_of_spurious_events_per_10_days=10, verbose=True
)

sensor_generator.fit(true_bg_trace)
plt.plot(sensor_generator.icgm_traces_used_in_training[0].T)
plt.show()

sensor_generator.percent_pass
sensor_generator.individual_sensor_properties
