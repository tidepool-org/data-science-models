import numpy as np
from tidepool_data_science_models.models.icgm_sensor_generator_functions import create_dataset, generate_icgm_sensors


def test_that_the_correct_number_of_spurious_events_are_created():

    # create a flat fake dataset
    true_df, _ = create_dataset(kind="flat", N=288 * 10, time_interval=5, flat_value=140, random_seed=0,)
    true_bg_trace = true_df["value"].values

    dist_params = [1, 1, 1, 1]
    for r in range(20):
        icgm_trace, ind_sensor_properties = generate_icgm_sensors(
            true_bg_trace,
            dist_params,  # [a, b, mu, sigma]
            n_sensors=1,  # (suggest 100 for speed, 1000 for thoroughness)
            bias_type="percentage_of_value",  # (constant_offset, percentage_of_value)
            bias_drift_type="none",  # options (none, linear, random)
            noise_coefficient=0,  # (0 ~ 60dB, 5 ~ 36 dB, 10, 30 dB)
            delay=0,  # (suggest 0, 5, 10, 15)
            number_of_spurious_events_per_10_days=r,
            random_seed=r,
        )

        icgm_true_diff = icgm_trace - true_bg_trace
        n_spurious_sensed = np.sum(np.log(abs(np.mean(icgm_true_diff) - icgm_true_diff)) > 1)
        assert n_spurious_sensed == r
