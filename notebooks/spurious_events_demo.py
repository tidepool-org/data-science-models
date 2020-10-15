import numpy as np
import matplotlib.pyplot as plt
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
from tidepool_data_science_models.models.icgm_sensor_generator_functions import create_dataset, generate_icgm_sensors

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
# first confirm that if number of spurious events is set to zero, that there are not spurious events
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
number_of_spurious_events_per_10_days = 1

for r in range(10):
    icgm_trace, ind_sensor_properties = generate_icgm_sensors(
        true_bg_trace,
        dist_params,  # [a, b, mu, sigma]
        n_sensors=1,  # (suggest 100 for speed, 1000 for thoroughness)
        bias_type="percentage_of_value",  # (constant_offset, percentage_of_value)
        bias_drift_type="none",  # options (none, linear, random)
        bias_drift_range=[0.85, 1.15],  # (suggest keeping within +/-15%)
        bias_drift_oscillations=1,  # opt for random drift (max of 2)
        noise_coefficient=2.5,  # (0 ~ 60dB, 5 ~ 36 dB, 10, 30 dB)
        delay=10,  # (suggest 0, 5, 10, 15)
        number_of_spurious_events_per_10_days=number_of_spurious_events_per_10_days,
        random_seed=r,
    )
    plt.plot(icgm_trace.T)
    plt.show()

# %% visually confirm the number of spurious events is correct
for r in range(10):
    icgm_trace, ind_sensor_properties = generate_icgm_sensors(
        true_bg_trace,
        dist_params,  # [a, b, mu, sigma]
        n_sensors=1,  # (suggest 100 for speed, 1000 for thoroughness)
        bias_type="percentage_of_value",  # (constant_offset, percentage_of_value)
        bias_drift_type="none",  # options (none, linear, random)
        bias_drift_range=[0.85, 1.15],  # (suggest keeping within +/-15%)
        bias_drift_oscillations=1,  # opt for random drift (max of 2)
        noise_coefficient=2.5,  # (0 ~ 60dB, 5 ~ 36 dB, 10, 30 dB)
        delay=10,  # (suggest 0, 5, 10, 15)
        number_of_spurious_events_per_10_days=r,
        random_seed=0,
    )
    plt.plot(icgm_trace.T)
    plt.show()


# %% visually confirm the number of spurious events per sensor is correct
for r in range(10):
    n_sensors = r + 1
    icgm_trace, ind_sensor_properties = generate_icgm_sensors(
        true_bg_trace,
        dist_params,  # [a, b, mu, sigma]
        n_sensors=n_sensors,  # (suggest 100 for speed, 1000 for thoroughness)
        bias_type="percentage_of_value",  # (constant_offset, percentage_of_value)
        bias_drift_type="none",  # options (none, linear, random)
        bias_drift_range=[0.85, 1.15],  # (suggest keeping within +/-15%)
        bias_drift_oscillations=1,  # opt for random drift (max of 2)
        noise_coefficient=2.5,  # (0 ~ 60dB, 5 ~ 36 dB, 10, 30 dB)
        delay=10,  # (suggest 0, 5, 10, 15)
        number_of_spurious_events_per_10_days=r,
        random_seed=0,
    )
    plt.plot(icgm_trace.T)
    plt.show()

# %% check to make sure that iCGM criteria is met
"""
(H) When iCGM values are less than 70 mg/dL, no corresponding blood glucose value shall read above 180 mg/dL.
(I) When iCGM values are greater than 180 mg/dL, no corresponding blood glucose value shall read less than 70 mg/dL.
"""
for s in range(n_sensors):
    icgm_sensor = icgm_trace[s, :]

    lt_70_mask = icgm_sensor < 70
    assert np.sum(true_bg_trace[lt_70_mask] > 180) == 0

    gt_180_mask = icgm_sensor > 180
    assert np.sum(true_bg_trace[gt_180_mask] < 70) == 0

# %% test to make sure that iCGM fit still works before applying spurious events
batch_training_size = 30
sensor_generator_1 = iCGMSensorGenerator(batch_training_size=batch_training_size, verbose=True)
sensor_generator_1.fit(true_bg_trace)
for b in range(batch_training_size):
    plt.plot(true_bg_trace)
    plt.plot(sensor_generator_1.icgm_traces_used_in_training[b])
    plt.show()

# %% 1 spurious event
batch_training_size = 30
sensor_generator_2 = iCGMSensorGenerator(
    batch_training_size=batch_training_size, max_number_of_spurious_events_per_10_days=20, verbose=True
)
sensor_generator_2.fit(true_bg_trace)
for b in range(batch_training_size):
    plt.plot(true_bg_trace)
    plt.plot(sensor_generator_2.icgm_traces_used_in_training[b])
    plt.show()


# %% check to make sure that iCGM criteria is met
"""
(H) When iCGM values are less than 70 mg/dL, no corresponding blood glucose value shall read above 180 mg/dL.
(I) When iCGM values are greater than 180 mg/dL, no corresponding blood glucose value shall read less than 70 mg/dL.
"""
icgm_traces = sensor_generator_2.icgm_traces_used_in_training
for s in range(n_sensors):
    icgm_sensor = icgm_traces[s, :]

    lt_70_mask = icgm_sensor < 70
    assert np.sum(true_bg_trace[lt_70_mask] > 180) == 0

    gt_180_mask = icgm_sensor > 180
    assert np.sum(true_bg_trace[gt_180_mask] < 70) == 0
