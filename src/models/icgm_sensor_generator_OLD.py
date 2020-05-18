#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ORIGINAL FILE: icgm_simulator.py
SOURCE COMMIT: https://github.com/tidepool-org/icgm-sensitivity-analysis/commit/fe82eefe4c991ec44c81d8c8ee0b9768759451a9

Created on Mon Nov 18 11:03:31 2019
@author: ed nykaza, jason meno

The Dexcom G6 Specifications in this file are publicly available from:
    “EVALUATION OF AUTOMATIC CLASS III DESIGNATION FOR
    Dexcom G6 Continuous Glucose Monitoring System.” n.d.
    https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN170088.pdf.

"""

# %% Libraries
import pandas as pd
import numpy as np
from scipy.optimize import brute, fmin

import src.models.icgm_sensor_functions_OLD as sf


# %% Functions
def icgm_simulator_old(
    SPECIAL_CONTROLS_CRITERIA_THRESHOLDS=[
        0.85,
        0.70,
        0.80,
        0.98,
        0.99,
        0.99,
        0.87,
    ],  # This is required only for iCGM sensors for now (A-G)
    n_sensors=3,
    use_g6_accuracy_in_loss=False,
    bias_type="percentage_of_value",
    bias_drift_type="random",
    random_seed=0,
    verbose=False,
    true_bg_trace=[],
    true_dataset_name="default",
):

    # pick delay based upon data in:
    # Vettoretti et al., 2019, Sensors 2019, 19, 5320
    if use_g6_accuracy_in_loss:
        delay = 5  # time delay between iCGM value and true value
    else:
        delay = 10  # based upon

    # set the random seed for reproducibility
    np.random.seed(seed=random_seed)

    """ capture settings """
    input_settings_table = sf.capture_settings(
        n_sensors, use_g6_accuracy_in_loss, bias_type, bias_drift_type, delay, random_seed
    )

    # STEP 1 STARTING TRUE TRACE
    # Default behavior of a sensor generating function should fail if no data is given
    # Explicitly call outside of this function to create a dataset if that's what we want
    if len(true_bg_trace) == 0:
        print("NO BG TRACE GIVEN! \n Creating 48 hour sinusoid dataset.")
        true_dataset_name = "48hours-sinusoid"
        true_df, true_df_inputs = sf.create_dataset(
            kind="sine",
            N=288 * 2,
            min_value=40,
            max_value=400,
            time_interval=5,
            flat_value=np.nan,
            oscillations=2,
            random_seed=random_seed,
        )
        true_bg_trace = np.array(true_df["value"])

    input_names = [
        "TRUE.kind",
        "TRUE.N",
        "TRUE.min_value",
        "TRUE.max_value",
        "TRUE.time_interval",
    ]

    true_df_inputs = pd.DataFrame(
        [true_dataset_name, len(true_bg_trace), np.min(true_bg_trace), np.max(true_bg_trace), 5],
        columns=["icgmSensorResults"],
        index=input_names,
    )

    # STEP 2 get initial johnsonsu distribution parameters AND search range
    # TODO: this should be an option if the johnsonsu parameters are not given
    # print("making {} sensors that meet the following specs:".format(n_sensors))
    # print("Criterion A-G = {} ".format(SPECIAL_CONTROLS_CRITERIA_THRESHOLDS))

    johnson_parameter_search_range, search_range_inputs = sf.get_search_range()

    sensor_results = brute(
        sf.johnsonsu_icgm_sensor,
        johnson_parameter_search_range,
        args=(
            true_bg_trace,
            SPECIAL_CONTROLS_CRITERIA_THRESHOLDS,
            n_sensors,
            bias_type,
            bias_drift_type,
            delay,
            random_seed,
            verbose,
            use_g6_accuracy_in_loss,
        ),
        workers=-1,
        full_output=True,
        finish=fmin,  # fmin will look for a local minimum around the grid point
    )

    # GET SENSOR METAPARAMETERS
    dist_params = sensor_results[0]

    (
        a,
        b,
        mu,
        sigma,
        noise_coefficient,
        bias_drift_range_min,
        bias_drift_range_max,
        bias_drift_oscillations,
    ) = dist_params
    bias_drift_range = [bias_drift_range_min, bias_drift_range_max]

    # STEP 3 apply the results
    icgm_traces, individual_sensor_properties = sf.generate_icgm_sensors(
        true_bg_trace,
        dist_params=dist_params[:4],
        n_sensors=n_sensors,
        bias_type=bias_type,
        bias_drift_type=bias_drift_type,
        bias_drift_range=bias_drift_range,
        bias_drift_oscillations=bias_drift_oscillations,
        noise_coefficient=noise_coefficient,
        delay=delay,
        random_seed=random_seed,
    )

    # using new (refactored) metrics
    df = sf.preprocess_data(true_bg_trace, icgm_traces, icgm_range=[40, 400], ysi_range=[0, 900])

    """ icgm special controls """
    icgm_special_controls_table = sf.calc_icgm_sc_table(df, "generic")

    """ new loss function """
    g6_loss, g6_table = sf.calc_dexcom_loss(df, n_sensors)
    if not use_g6_accuracy_in_loss:
        g6_loss = np.nan

    loss_score, percent_pass = sf.calc_icgm_special_controls_loss(icgm_special_controls_table, g6_loss)

    """ overall results """
    overall_metrics_table = sf.calc_overall_metrics(df)
    overall_metrics_table.loc["ICGM_PASS%", "icgmSensorResults"] = percent_pass
    overall_metrics_table.loc["LOSS_SCORE", "icgmSensorResults"] = loss_score

    # Get individual sensor special controls results
    trace_len = len(true_bg_trace)
    sensor_nPairs = []
    sensor_icgmSensorResults = []
    sensor_metrics = []
    for i in range(n_sensors):
        ind_sensor_df = df.iloc[trace_len * i : trace_len * (i + 1)]
        ind_sensor_special_controls_table = sf.calc_icgm_sc_table(ind_sensor_df, "generic")

        loss_score, percent_pass = sf.calc_icgm_special_controls_loss(ind_sensor_special_controls_table, g6_loss)

        sensor_metrics_table = sf.calc_overall_metrics(ind_sensor_df)
        sensor_metrics_table.loc["ICGM_PASS%", "icgmSensorResults"] = percent_pass
        sensor_metrics_table.loc["LOSS_SCORE", "icgmSensorResults"] = loss_score
        # sensor_metrics_table
        sensor_nPairs.append(ind_sensor_special_controls_table["nPairs"].values)
        sensor_icgmSensorResults.append(ind_sensor_special_controls_table["icgmSensorResults"].values)
        sensor_metrics.append(sensor_metrics_table.T)

    sensor_nPair_cols = icgm_special_controls_table.T.add_suffix("_nPairs").columns

    sensor_results_cols = icgm_special_controls_table.T.add_suffix("_results").columns

    sensor_nPairs = pd.DataFrame(sensor_nPairs, columns=sensor_nPair_cols)
    sensor_icgmSensorResults = pd.DataFrame(sensor_icgmSensorResults, columns=sensor_results_cols)
    ind_sensor_metrics = pd.concat(sensor_metrics).reset_index(drop=True)
    individual_sensor_properties.reset_index(drop=True, inplace=True)

    individual_sensor_properties = pd.concat(
        [individual_sensor_properties, ind_sensor_metrics, sensor_nPairs, sensor_icgmSensorResults], axis=1
    )

    dist_param_names = [
        "a",
        "b",
        "mu",
        "sigma",
        "batch_noise_coefficient",
        "bias_drift_range_min",
        "bias_drift_range_max",
        "batch_bias_drift_oscillations",
    ]

    dist_df = pd.DataFrame(dist_params, columns=["icgmSensorResults"], index=dist_param_names)

    dist_df.loc["bias_drift_type"] = bias_drift_type

    """ dexcom g6 accuracy metric (tables)"""
    # gsc = sf.calc_icgm_sc_table(df, "g6")
    # g1a = sf.calc_g6_table1A(df, n_sensors)
    # g1b = sf.calc_g6_table1BF(df, n_sensors, "B")
    # g1f = sf.calc_g6_table1BF(df, "F")
    # g3a = sf.calc_g6_table3AC(df, n_sensors, "A")
    # g3c = sf.calc_g6_table3AC(df, "C")
    # g4 = sf.calc_g6_table4(df, n_sensors)
    # g6 = sf.calc_g6_table6(df, n_sensors)

    results_df = pd.concat(
        [input_settings_table, true_df_inputs, search_range_inputs, dist_df, overall_metrics_table, g6_table],
        sort=False,
    )

    batch_sensor_properties = results_df[~results_df.index.duplicated(keep="first")]
    sc_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    batch_sensor_properties.drop(index=sc_letters, inplace=True)
    batch_sensor_properties.drop(columns=["dexG6"], inplace=True)

    batch_sc_table = icgm_special_controls_table

    batch_sc_npairs = pd.DataFrame(batch_sc_table["nPairs"].T.add_suffix("_nPairs"))
    batch_sc_npairs.columns = ["icgmSensorResults"]

    batch_sc_results = pd.DataFrame(batch_sc_table["icgmSensorResults"].T.add_suffix("_results"))

    batch_sensor_properties = pd.concat([batch_sensor_properties, batch_sc_npairs, batch_sc_results])

    return icgm_traces, individual_sensor_properties, batch_sensor_properties
