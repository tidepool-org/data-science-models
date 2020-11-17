import os
import datetime
import numpy as np
import pandas as pd
from tidepool_data_science_models.models.icgm_sensor_generator_functions import (
    calc_icgm_sc_table,
    preprocess_data,
    calc_mard,
    calc_mbe,
)


def calc_percent_meeting_sc_criterion(true_ndarray, icgm_ndarray, icgm_range=[40, 400]):

    within_measurement_range = (icgm_ndarray >= icgm_range[0]) & (icgm_ndarray <= icgm_range[1])
    true_bg = true_ndarray[within_measurement_range]
    icgm_bg = icgm_ndarray[within_measurement_range]

    within_15_mg_dl = (true_bg - 15 <= icgm_bg) & (icgm_bg <= true_bg + 15)
    within_40_mg_dl = (true_bg - 40 <= icgm_bg) & (icgm_bg <= true_bg + 40)

    within_15_percent = (true_bg * 0.85 <= icgm_bg) & (icgm_bg <= true_bg * 1.15)
    within_40_percent = (true_bg * 0.60 <= icgm_bg) & (icgm_bg <= true_bg * 1.40)

    within_20_percent = (true_bg * 0.80 <= icgm_bg) & (icgm_bg <= true_bg * 1.20)

    icgm_lt_70 = icgm_bg < 70
    icgm_bt_70_180 = (icgm_bg >= 70) & (icgm_bg <= 180)
    icgm_gt_180 = icgm_bg > 180

    true_lt_70 = true_bg < 70
    true_gt_180 = true_bg > 180

    n_icgm_lt_70 = icgm_lt_70.sum()
    n_icgm_bt_70_180 = icgm_bt_70_180.sum()
    n_icgm_gt_180 = icgm_gt_180.sum()
    n_within_measurement_range = within_measurement_range.sum()

    # get rates
    true_rate_ndarray = (true_ndarray - np.roll(true_ndarray, 1)) / 5
    true_rate_ndarray[:, 0] = 0
    true_rate_gt2 = true_rate_ndarray > 2
    true_rate_lt_neg2 = true_rate_ndarray < -2

    icgm_rate_ndarray = (icgm_ndarray - np.roll(icgm_ndarray, 1)) / 5
    icgm_rate_ndarray[:, 0] = 0

    icgm_rate_gt1 = icgm_rate_ndarray > 1
    n_icgm_rate_gt1 = icgm_rate_gt1.sum()

    icgm_rate_lt_neg1 = icgm_rate_ndarray < -1
    n_icgm_rate_lt_neg1 = icgm_rate_lt_neg1.sum()

    # Special Controls Criteria
    criterion_A = within_15_mg_dl[icgm_lt_70].sum() / n_icgm_lt_70
    criterion_B = within_15_percent[icgm_bt_70_180].sum() / n_icgm_bt_70_180
    criterion_C = within_15_percent[icgm_gt_180].sum() / n_icgm_gt_180
    criterion_D = within_40_mg_dl[icgm_lt_70].sum() / n_icgm_lt_70
    criterion_E = within_40_percent[icgm_bt_70_180].sum() / n_icgm_bt_70_180
    criterion_F = within_40_percent[icgm_gt_180].sum() / n_icgm_gt_180
    criterion_G = within_20_percent.sum() / n_within_measurement_range
    criterion_H = 100 - ((np.sum((icgm_lt_70) & (true_gt_180)) / n_icgm_lt_70) * 100)
    criterion_I = 100 - ((np.sum((icgm_gt_180) & (true_lt_70)) / n_icgm_gt_180) * 100)
    criterion_J = 100 - ((np.sum((icgm_rate_gt1) & (true_rate_lt_neg2)) / n_icgm_rate_gt1) * 100)
    criterion_K = 100 - ((np.sum((icgm_rate_lt_neg1) & (true_rate_gt2)) / n_icgm_rate_lt_neg1) * 100)

    percent_meeting_criterion = [
        criterion_A,
        criterion_B,
        criterion_C,
        criterion_D,
        criterion_E,
        criterion_F,
        criterion_G,
        criterion_H,
        criterion_I,
        criterion_J,
        criterion_K,
    ]

    n_per_region = [
        n_icgm_lt_70,
        n_icgm_bt_70_180,
        n_icgm_gt_180,
        n_within_measurement_range,
        n_icgm_rate_gt1,
        n_icgm_rate_lt_neg1,
    ]

    return percent_meeting_criterion, n_per_region


def bootstrap_method(true_ndarray, icgm_ndarray, n_bootstrap_samples=50):
    n_sensors, n_datapoints = np.shape(true_ndarray)

    bootstrap_results = np.zeros([11, n_bootstrap_samples]) * np.nan
    n_pairs = np.zeros([6, n_bootstrap_samples]) * np.nan

    for i in range(0, n_bootstrap_samples):
        sampled_sensor_ids = np.random.choice(n_sensors, n_sensors, replace=True)
        bootstrap_results[:, i], n_pairs[:, i] = calc_percent_meeting_sc_criterion(
            true_ndarray[sampled_sensor_ids, :], icgm_ndarray[sampled_sensor_ids, :]
        )

    bootstrap_lb_df = pd.DataFrame(
        np.percentile(bootstrap_results[0:7, :], 0.05 * 100, 1) * 100,
        index=["A", "B", "C", "D", "E", "F", "G"],
        columns=["meetBootstrap"],
    )

    bootstrap_lb_df.loc["H", "meetBootstrap"] = bootstrap_results[7, :].min()
    bootstrap_lb_df.loc["I", "meetBootstrap"] = bootstrap_results[8, :].min()

    total_icgm_lt_1 = n_pairs[4, :].sum()
    total_icgm_gt_neg1 = n_pairs[5, :].sum()

    bootstrap_lb_df.loc["J", "meetBootstrap"] = (
        np.sum((bootstrap_results[9, :] / 100) * n_pairs[4, :]) / total_icgm_lt_1 * 100
    )
    bootstrap_lb_df.loc["K", "meetBootstrap"] = (
        np.sum((bootstrap_results[10, :] / 100) * n_pairs[5, :]) / total_icgm_gt_neg1 * 100
    )

    bootstrap_lb_df["bootstrap_pairs"] = np.nan
    bootstrap_lb_df.loc[["A", "D"], "bootstrap_pairs"] = n_pairs[0, :].mean()
    bootstrap_lb_df.loc[["B", "E"], "bootstrap_pairs"] = n_pairs[1, :].mean()
    bootstrap_lb_df.loc[["C", "F"], "bootstrap_pairs"] = n_pairs[2, :].mean()
    bootstrap_lb_df.loc[["G"], "bootstrap_pairs"] = n_pairs[3, :].mean()
    bootstrap_lb_df.loc[["H"], "bootstrap_pairs"] = n_pairs[0, :].sum()
    bootstrap_lb_df.loc[["I"], "bootstrap_pairs"] = n_pairs[2, :].sum()
    bootstrap_lb_df.loc[["J"], "bootstrap_pairs"] = n_pairs[4, :].sum()
    bootstrap_lb_df.loc[["K"], "bootstrap_pairs"] = n_pairs[5, :].sum()

    return bootstrap_lb_df


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

batch_size = 324
n_smoothing_points = 3
dataset_range = range(0, 10)
subset_size = 10
n_points_in_subset = 77
n_bootstrap_samples = 500
datetime_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

file_name = "compare_bootstrap_with_binomial-n_sensors_{}-n_ysi_{}-n_bootstrap_samples_{}-{}.csv".format(
    batch_size, n_points_in_subset, n_bootstrap_samples, datetime_now
)

all_diff_results = pd.DataFrame(
    columns=["dataset_id", "subset_id", "noise_sigma", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
)

idx = 0
for dataset_id in dataset_range:  # range(0, 10):
    tbddp_bg_chain_df = pd.read_csv(
        os.path.join("..", "data", "tbddp_bg_chains", "bg_{}.csv".format(dataset_id)), index_col=[0]
    )

    for subset in range(subset_size):

        start_index = np.random.randint(len(tbddp_bg_chain_df) - n_points_in_subset)
        end_index = start_index + n_points_in_subset
        bg_df = tbddp_bg_chain_df[start_index:end_index].copy()

        true_bg_array = bg_df["bg"].rolling(n_smoothing_points, center=True).mean().dropna().values
        bg_rates = (true_bg_array[1:] - true_bg_array[:-1]) / 5

        # %% add noise to true bg values
        n_values = len(true_bg_array)
        true_matrix = np.tile(true_bg_array, (batch_size, 1))
        true_array = true_matrix.flatten()

        for noise_sigma in range(8, 10, 1):

            icgm_matrix = true_matrix + np.random.normal(loc=0, scale=noise_sigma, size=(batch_size, n_values))
            icgm_array = icgm_matrix.flatten()
            # remove any values outside of measurement range
            within_measurement_range = (icgm_array >= 40) & (icgm_array <= 400)
            icgm_error = icgm_array[within_measurement_range] - true_array[within_measurement_range]
            abs_difference_error = np.abs(icgm_error)
            abs_relative_difference = 100 * abs_difference_error / true_array[within_measurement_range]
            mean_absolute_relative_difference = np.mean(abs_relative_difference)

            bg_df = preprocess_data(true_bg_array, icgm_matrix, icgm_range=[40, 400], ysi_range=[0, 900])
            batch_mard = calc_mard(bg_df)
            batch_mbe = calc_mbe(bg_df)
            acc_results = calc_icgm_sc_table(bg_df)

            # %% bootstrapped results
            bootstrap_df = bootstrap_method(true_matrix, icgm_matrix, n_bootstrap_samples=n_bootstrap_samples)
            all_results = pd.merge(acc_results, bootstrap_df, left_index=True, right_index=True)

            bootstrap_diff = all_results["meetBootstrap"] - all_results["icgmSensorResults"]
            all_diff_results.loc[idx, "dataset_id"] = dataset_id
            all_diff_results.loc[idx, "subset_id"] = subset
            all_diff_results.loc[idx, "noise_sigma"] = noise_sigma
            all_diff_results.loc[idx, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]] = bootstrap_diff.values
            idx = idx + 1

            print(dataset_id, subset, noise_sigma, all_results[["icgmSensorResults", "meetBootstrap"]])

all_diff_not_null = all_diff_results.loc[:, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]].dropna().values
all_diff_summary_min = all_diff_not_null.flatten().min()
all_diff_summary_median = np.median(all_diff_not_null.flatten())
all_diff_summary_max = all_diff_not_null.flatten().max()
all_diff_summary_mean = all_diff_not_null.flatten().mean()
all_diff_summary_std = all_diff_not_null.flatten().std()
print("saving {}".format(file_name))
print("summary min-median-max={} {} {}".format(all_diff_summary_min, all_diff_summary_median, all_diff_summary_max))
print("all_diff_summary_mean={} +/- {}".format(all_diff_summary_mean, all_diff_summary_std))

all_diff_results.to_csv(file_name)