import os
import datetime
import numpy as np
import pandas as pd
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
from tidepool_data_science_models.models.icgm_sensor_generator_functions import (
    calc_icgm_special_controls_loss,
    calc_icgm_sc_table,
    generate_icgm_sensors,
    preprocess_data,
)

experiment_results = pd.DataFrame(
    columns=[
        "dataset_id",
        "dataset_size",
        "batch_size",
        "n_smoothing_points",
        "percent_pass",
        "a",
        "b",
        "mu",
        "sigma",
        "max_noise",
        "drift_min",
        "drift_max",
        "drift_osc",
    ]
)

batch_size = 100
n_smoothing_points = 3
dataset_range = range(0, 10)
for dataset_id in dataset_range:  # range(0, 10):
    bg_df = pd.read_csv(os.path.join("..", "data", "tbddp_bg_chains", "bg_{}.csv".format(dataset_id)), index_col=[0])

    true_bg_array = bg_df["bg"].rolling(n_smoothing_points, center=True).mean().dropna().values
    bg_rates = (true_bg_array[1:] - true_bg_array[:-1]) / 5
    print(bg_rates.min(), bg_rates.mean(), bg_rates.max())
    # true_bg_array = bg_df["bg"].values

    sensor_generator = iCGMSensorGenerator(
        batch_training_size=batch_size,
        # max_number_of_spurious_events_per_10_days=0,
        verbose=True
    )

    sensor_generator.fit(true_bg_array)

    print(sensor_generator.percent_pass)
    print(sensor_generator.icgm_special_controls_accuracy_table)
    print(sensor_generator.johnson_parameter_search_range)
    print(sensor_generator.dist_params)
    sensor_generator.individual_sensor_properties[1:2].T

    experiment_results.loc[dataset_id, "dataset_id"] = dataset_id
    experiment_results.loc[dataset_id, "dataset_size"] = len(bg_df)
    experiment_results.loc[dataset_id, "batch_size"] = batch_size
    experiment_results.loc[dataset_id, "n_smoothing_points"] = n_smoothing_points
    experiment_results.loc[dataset_id, "percent_pass"] = sensor_generator.percent_pass
    experiment_results.loc[
        dataset_id, ["a", "b", "mu", "sigma", "max_noise", "drift_min", "drift_max", "drift_osc"]
    ] = sensor_generator.dist_params

    test_set = set([*dataset_range]) - set([dataset_id])
    for test_dataset_id in test_set:
        bg_df_test = pd.read_csv(
            os.path.join("..", "data", "tbddp_bg_chains", "bg_{}.csv".format(test_dataset_id)), index_col=[0]
        )

        test_bg_trace = bg_df_test["bg"].rolling(n_smoothing_points, center=True).mean().dropna().values
        delayed_iCGM, ind_sensor_properties = generate_icgm_sensors(
            test_bg_trace,
            sensor_generator.dist_params[0:4],  # [a, b, mu, sigma]
            n_sensors=batch_size,  # (suggest 100 for speed, 1000 for thoroughness)
            bias_type="percentage_of_value",  # (constant_offset, percentage_of_value)
            bias_drift_type="random",  # options (none, linear, random)
            bias_drift_range=sensor_generator.dist_params[5:7],  # (suggest keeping within +/-15%)
            bias_drift_oscillations=sensor_generator.dist_params[7],  # opt for random drift (max of 2)
            noise_coefficient=sensor_generator.dist_params[4],  # (0 ~ 60dB, 5 ~ 36 dB, 10, 30 dB)
            delay=10,  # (suggest 0, 5, 10, 15)
            # number_of_spurious_events_per_10_days=0,
            random_seed=0,
        )

        acc_results = calc_icgm_sc_table(
            preprocess_data(test_bg_trace, delayed_iCGM, icgm_range=[40, 400], ysi_range=[0, 900])
        )
        _, percent_pass_test = calc_icgm_special_controls_loss(acc_results, np.nan)
        print(test_dataset_id, percent_pass_test)
        print(acc_results)
        experiment_results.loc[dataset_id, "test_{}".format(test_dataset_id)] = percent_pass_test

experiment_results.to_csv("train_test_icgm_results-{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")))