import numpy as np
from tidepool_data_science_models.models.icgm_sensor_generator_functions import (
    create_dataset,
    generate_icgm_sensors,
    preprocess_data,
    calc_icgm_sc_table,
)


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


def test_that_spurious_event_values_do_not_exceed_special_control_limits_h_and_i():
    """
    (H) When iCGM values are less than 70 mg/dL, no corresponding blood glucose value shall read above 180 mg/dL.
    (I) When iCGM values are greater than 180 mg/dL, no corresponding blood glucose value shall read less than 70 mg/dL.
    """
    true_df, _ = create_dataset(kind="linear", N=288 * 10, min_value=40, max_value=400, time_interval=5, random_seed=0,)
    true_bg_trace = true_df["value"].values

    icgm_trace, ind_sensor_properties = generate_icgm_sensors(
        true_bg_trace,
        [1, 1, 1, 1],  # [a, b, mu, sigma]
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

    """
    (H) When iCGM values are less than 70 mg/dL, no corresponding blood glucose value shall read above 180 mg/dL.
    (I) When iCGM values are greater than 180 mg/dL, no corresponding blood glucose value shall read less than 70 mg/dL.
    """

    icgm_special_controls_table = calc_icgm_sc_table(
        preprocess_data(true_bg_trace, icgm_trace, icgm_range=[40, 400], ysi_range=[0, 900])
    )

    assert icgm_special_controls_table.loc["H", "icgmSensorResults"] == 100
    assert icgm_special_controls_table.loc["I", "icgmSensorResults"] == 100
