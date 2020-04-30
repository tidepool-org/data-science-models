#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
import os
import datetime
from scipy.optimize import brute, fmin

from scipy.stats import johnsonsu
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import icgm_sensor_functions_OLD as sf


# %% Functions
def icgm_simulator(SPECIAL_CONTROLS_CRITERIA_THRESHOLDS=[0.85,   # A
                                                         0.70,   # B
                                                         0.80,   # C
                                                         0.98,   # D
                                                         0.99,   # E
                                                         0.99,   # F
                                                         0.87],  # G
                   n_sensors=30,
                   use_g6_accuracy_in_loss=False,
                   bias_type="percentage_of_value",
                   bias_drift_type="random",
                   random_seed=0,
                   verbose=False,
                   save_results=False,
                   make_figures=False,
                   true_bg_trace=[],
                   true_dataset_name="default"):

    # pick delay based upon data in:
    # Vettoretti et al., 2019, Sensors 2019, 19, 5320
    if use_g6_accuracy_in_loss:
        delay = 5  # time delay between iCGM value and true value
    else:
        delay = 10  # based upon

    # set the random seed for reproducibility
    np.random.seed(seed=random_seed)

    ''' capture settings '''
    input_settings_table = sf.capture_settings(
        n_sensors,
        use_g6_accuracy_in_loss,
        bias_type,
        bias_drift_type,
        delay,
        random_seed
    )

    # STEP 1 STARTING TRUE TRACE
    if len(true_bg_trace) == 0:
        print("NO BG TRACE GIVEN! \n Creating 48 hour sinusoid dataset.")
        true_dataset_name = "48hours-sinusoid"
        true_df, true_df_inputs = sf.create_dataset(
                kind="sine",
                N=288*2,
                min_value=40,
                max_value=400,
                time_interval=5,
                flat_value=np.nan,
                oscillations=2,
                random_seed=random_seed
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
        [
            true_dataset_name,
            len(true_bg_trace),
            np.min(true_bg_trace),
            np.max(true_bg_trace),
            5,
        ],
        columns=["icgmSensorResults"],
        index=input_names
    )

    # STEP 2 get initial johnsonsu distribution parameters AND search range
    # TODO: this should be an option if the johnsonsu parameters are not given
    # print("making {} sensors that meet the following specs:".format(n_sensors))
    # print("Criterion A-G = {} ".format(SPECIAL_CONTROLS_CRITERIA_THRESHOLDS))

    johnson_parameter_search_range, search_range_inputs = \
        sf.get_search_range()

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
             use_g6_accuracy_in_loss
         ),
         workers=-1,
         full_output=True,
         finish=fmin  # fmin will look for a local minimum around the grid point
     )

    # GET SENSOR METAPARAMETERS
    dist_params = sensor_results[0]

    # HARDCODED METAPARAMETERS TO FIT METABOLIC SIMULATOR
    # ORANGE 100 24-hour seed 2020-04-03
#    dist_params = [
#        0.453367301,  # a
#        22.7526517,  # b
#        5.94849642,  # mu
#        32.4910887,  # sigma
#        8.91309338,  # noise_coefficient
#        0.951195936,  # bias_drift_range_min
#        1.01342805,  # bias_drift_range_max
#        -0.00135629247   # bias_drift_oscillations
#    ]


    # BLUE-100 - High accuracy, based on 8-hour simulation data
#    dist_params = [
#        0.7777797779024997,  # a
#        16.000366070969612,  # b
#        3.994903813972459,  # mu
#        49.507627726706794,  # sigma
#        8.540940003169371,  # noise_coefficient
#        1.0340191859896493,  # bias_drift_range_min
#        1.0078798143881644,  # bias_drift_range_max
#        -0.00026210704069611394   # bias_drift_oscillations
#    ]

    # ORANGE-100 - Low accuracy, based on 10-day seed data
#    dist_params = [
#        0.7499006738942093,  # a
#        23.180403009646206,  # b
#        4.263250748156536,  # mu
#        44.66613582316239,  # sigma
#        2.447144587339765,  # noise_coefficient
#        0.8409223049247742,  # bias_drift_range_min
#        1.0741877708314456,  # bias_drift_range_max
#        0.8899017649283527   # bias_drift_oscillations
#    ]


    # ORANGE-30 - Low accuracy, based on 10-day seed data
#    dist_params = [
#        0.545875,  # a
#        20.637930,  # b
#        4.233619,  # mu
#        92.405443,  # sigma
#        3.180765,  # noise_coefficient
#        1.0073474196487227,  # bias_drift_range_min
#        1.0371552026227717,  # bias_drift_range_max
#        0.0007532920030378398   # bias_drift_oscillations
#    ]

    # BLUE-30 High accuracy, based on 8-hour simulation data
#    dist_params = [
#        0.59012889,  # a
#        22.73508938,  # b
#        2.93902946,  # mu
#        51.16303496,  # sigma
#        2.380271112074799,  # noise_coefficient
#        1.0073474196487227,  # bias_drift_range_min
#        1.0371552026227717,  # bias_drift_range_max
#        0.0007532920030378398   # bias_drift_oscillations
#    ]
    (
         a, b, mu, sigma, noise_coefficient,
         bias_drift_range_min, bias_drift_range_max, bias_drift_oscillations
    ) = dist_params
    bias_drift_range = [bias_drift_range_min, bias_drift_range_max]

    # loss = sensor_results[1]
    # iteration_parameters = sensor_results[2]
    # iteration_loss_scores = sensor_results[3]

    # capture the results of all of the brute force runs
    # iteration_results = pd.DataFrame(
    #     iteration_loss_scores.reshape([-1, 1]), columns=["loss"]
    # )
    # iteration_results["a"] = \
    #     iteration_parameters[0, :, :, :, :].reshape([-1, 1])
    # iteration_results["b"] = \
    #     iteration_parameters[1, :, :, :, :].reshape([-1, 1])
    # iteration_results["mu"] = \
    #     iteration_parameters[2, :, :, :, :].reshape([-1, 1])
    # iteration_results["sigma"] = (
    #     iteration_parameters[3, :, :, :, :].reshape([-1, 1])
    # )
    # iteration_results["noise_coefficient"] = (
    #     iteration_parameters[4, :, :, :, :].reshape([-1, 1])
    # )
    # iteration_results["bias_drift_range_min"] = (
    #     iteration_parameters[5, :, :, :, :].reshape([-1, 1])
    # )
    # iteration_results["bias_drift_range_max"] = (
    #     iteration_parameters[6, :, :, :, :].reshape([-1, 1])
    # )
    # iteration_results["bias_drift_oscillations"] = (
    #     iteration_parameters[7, :, :, :, :].reshape([-1, 1])
    # )

    # print("done\nthe sensor has the following distribution parameters:")
    # print("a={}, b={}, mu={}, sigma={}".format(
    #         np.round(dist_params[0], 1),
    #         np.round(dist_params[1], 1),
    #         np.round(dist_params[2], 1),
    #         np.round(dist_params[3], 1)
    # ))
    # print("the overall loss score was {}".format(np.round(loss, 4)))
    # print("a score close to 0 and less than 1 is good")

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

    # this part is used in iCGM
    for sensor_number in np.arange(0, n_sensors):
        sensor_values = \
            individual_sensor_properties.loc[sensor_number, :].values[0]
        true_value_at_time_t = sf.get_icgm_value(
            true_bg_value=true_bg_trace[0],
            at_time=0,
            random_seed=np.int(sensor_values[8]),
            initial_bias=sensor_values[0],
            phi_drift=sensor_values[1],
            bias_drift_range=[sensor_values[2], sensor_values[3]],
            bias_drift_oscillations=sensor_values[4],
            bias_norm_factor=sensor_values[5],
            noise_coefficient=sensor_values[6],
        )
        # print(true_value_at_time_t, true_bg_trace[0])

    # using new (refactored) metrics
    df = sf.preprocess_data(
            true_bg_trace,
            icgm_traces,
            icgm_range=[40, 400],
            ysi_range=[0, 900]
    )

    ''' icgm special controls '''
    icgm_special_controls_table = sf.calc_icgm_sc_table(df, "generic")

    ''' new loss function '''
    g6_loss, g6_table = sf.calc_dexcom_loss(df, n_sensors)
    if not use_g6_accuracy_in_loss:
        g6_loss = np.nan

    loss_score, percent_pass = (
        sf.calc_icgm_special_controls_loss(icgm_special_controls_table,
                                           g6_loss)
    )

    ''' overall results '''
    overall_metrics_table = sf.calc_overall_metrics(df)
    overall_metrics_table.loc["ICGM_PASS%", "icgmSensorResults"] = percent_pass
    overall_metrics_table.loc["LOSS_SCORE", "icgmSensorResults"] = loss_score

    # Get individual sensor special controls results
    trace_len = len(true_bg_trace)
    sensor_nPairs = []
    sensor_icgmSensorResults = []
    sensor_metrics = []
    for i in range(n_sensors):
        ind_sensor_df = df.iloc[trace_len*i:trace_len*(i+1)]
        ind_sensor_special_controls_table = (
                sf.calc_icgm_sc_table(ind_sensor_df, "generic")
        )

        loss_score, percent_pass = (
                sf.calc_icgm_special_controls_loss(
                        ind_sensor_special_controls_table,
                        g6_loss
                )
        )

        sensor_metrics_table = sf.calc_overall_metrics(ind_sensor_df)
        sensor_metrics_table.loc["ICGM_PASS%", "icgmSensorResults"] = (
                percent_pass
        )
        sensor_metrics_table.loc["LOSS_SCORE", "icgmSensorResults"] = (
                loss_score
        )
        # sensor_metrics_table
        sensor_nPairs.append(
                ind_sensor_special_controls_table['nPairs'].values
        )
        sensor_icgmSensorResults.append(
                ind_sensor_special_controls_table['icgmSensorResults'].values
        )
        sensor_metrics.append(sensor_metrics_table.T)

    sensor_nPair_cols = (
            icgm_special_controls_table.T.add_suffix('_nPairs').columns
    )

    sensor_results_cols = (
            icgm_special_controls_table.T.add_suffix('_results').columns
    )

    sensor_nPairs = pd.DataFrame(sensor_nPairs, columns=sensor_nPair_cols)
    sensor_icgmSensorResults = pd.DataFrame(
            sensor_icgmSensorResults, columns=sensor_results_cols
    )
    ind_sensor_metrics = pd.concat(sensor_metrics).reset_index(drop=True)
    individual_sensor_properties.reset_index(drop=True, inplace=True)

    individual_sensor_properties = pd.concat(
            [
                    individual_sensor_properties,
                    ind_sensor_metrics,
                    sensor_nPairs,
                    sensor_icgmSensorResults
            ], axis=1
    )

    dist_param_names = [
        "a", "b", "mu", "sigma", "batch_noise_coefficient",
        "bias_drift_range_min", "bias_drift_range_max",
        "batch_bias_drift_oscillations"
    ]

    dist_df = pd.DataFrame(
        dist_params,
        columns=["icgmSensorResults"],
        index=dist_param_names
    )

    dist_df.loc["bias_drift_type"] = bias_drift_type

    ''' dexcom g6 accuracy metric (tables)'''
    gsc = sf.calc_icgm_sc_table(df, "g6")
    g1a = sf.calc_g6_table1A(df, n_sensors)
    g1b = sf.calc_g6_table1BF(df, n_sensors, "B")
    g1f = sf.calc_g6_table1BF(df, "F")
    g3a = sf.calc_g6_table3AC(df, n_sensors, "A")
    g3c = sf.calc_g6_table3AC(df, "C")
    g4 = sf.calc_g6_table4(df, n_sensors)
    g6 = sf.calc_g6_table6(df, n_sensors)

    results_df = pd.concat([
        input_settings_table,
        true_df_inputs,
        search_range_inputs,
        dist_df,
        overall_metrics_table,
        g6_table,
    ], sort=False)

    loss_score = np.round(loss_score, 2)
    percent_pass = np.round(percent_pass)
    snr_db = np.round(overall_metrics_table.loc["SNR", "icgmSensorResults"])
    mard = np.round(overall_metrics_table.loc["MARD", "icgmSensorResults"], 2)
    mbe = np.round(overall_metrics_table.loc["MBE", "icgmSensorResults"], 2)

    file_name = (
        "{}-sensors-generated-on-{}".format(
            n_sensors, datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        )
        + "-from={}".format(true_dataset_name)
        + "-seed={}".format(random_seed)
        + "-G6={}".format(use_g6_accuracy_in_loss)
        + "-meet-icgm={}".format(percent_pass)
        + "-LOSS={}".format(loss_score)
        + "-MARD={}".format(mard)
        + "-MBE={}".format(mbe)
        + "-SNR={}".format(snr_db)
        + "-lag={}".format(delay)
    )

    batch_icgm_results = results_df[~results_df.index.duplicated(keep='first')]
    sc_letters = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    batch_icgm_results.drop(index=sc_letters, inplace=True)
    batch_icgm_results.drop(columns=['dexG6'], inplace=True)

    batch_sc_table = icgm_special_controls_table

    batch_sc_npairs = pd.DataFrame(
            batch_sc_table['nPairs'].T.add_suffix('_nPairs')
    )
    batch_sc_npairs.columns = ['icgmSensorResults']

    batch_sc_results = pd.DataFrame(
            batch_sc_table['icgmSensorResults'].T.add_suffix('_results')
    )

    batch_icgm_results = pd.concat(
            [
                    batch_icgm_results,
                    batch_sc_npairs,
                    batch_sc_results
            ]
    )


    return icgm_traces, individual_sensor_properties, batch_icgm_results


def make_plotly_figures(figure_metadata):
    """Makes two plotly figures: one of the johnson distribution and another
    for the true BG + iCGM traces
    """

    # TODO: Properly parse out the needed figure metadata
    (a, b, mu, sigma,
     loss_score, percent_pass,
     n_sensors,
     noise_coefficient,
     bias_drift_range,
     bias_drift_oscillations,
     overall_metrics_table,
     icgm_traces,
     true_bg_trace,
     SPECIAL_CONTROLS_CRITERIA_THRESHOLDS,
     bias_type,
     bias_drift_type) = figure_metadata

    # here is the cumulative distribtuion
    x = np.arange(-60, 61, 1)
    # cdf = johnsonsu.cdf(x, a=a, b=b, loc=mu, scale=sigma) * 100

    # title = (
    #     "Cumulative Distribution that Describes Sensor Performance<br>"
    #     + "Johnson SU Dist with a={}, b={}, mu={}, sigma={}".format(
    #         np.round(a, 1),
    #         np.round(b, 2),
    #         np.round(mu, 1),
    #         np.round(sigma, 1),
    #     )
    # )

    # cdf_fig = px.line(x=x, y=cdf, title=title)
    # cdf_fig.update_layout(
    #     xaxis_title_text='Starting Bias Error (mg/dL)',
    #     yaxis_title_text='Percent of Values Below (%)',
    #     xaxis_showgrid=True,
    #     xaxis_zeroline=False,
    #     yaxis_zeroline=False,
    # )

    # cdf_fig.update_xaxes(tickvals=[-50, -40, -20, -15, 0, 15, 20, 40, 50])
    # cdf_fig.update_yaxes(tickvals=[1, 5, 10, 25, 50, 75, 90, 95, 99])
    # plot(cdf_fig)

    # plot the pdf
    title = (
        "Probability Density Function that Describes Sensor Performance<br>"
        + "Johnson SU Dist with a={}, b={}, mu={}, sigma={}".format(
            np.round(a, 1),
            np.round(b, 2),
            np.round(mu, 1),
            np.round(sigma, 1),
        )
    )

    pdf = johnsonsu.pdf(x, a=a, b=b, loc=mu, scale=sigma)

    pdf_df = pd.DataFrame([x, pdf]).T
    pdf_df.columns = ['x', 'pdf']

    pdf_fig = px.line(pdf_df, x='x', y='pdf', title=title)
    pdf_fig.update_layout(
        xaxis_title_text='Starting Bias Error (mg/dL)',
        yaxis_title_text='Probability Density',
        xaxis_showgrid=True,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
    )

    pdf_fig.update_xaxes(tickvals=[-50, -40, -20, -15, 0, 15, 20, 40, 50])
    plot(pdf_fig)

    # plot the traces
    loss_score = np.round(loss_score, 3)
    percent_pass = np.round(percent_pass)
    snr_db = np.round(overall_metrics_table.loc["SNR", "icgmSensorResults"])
    mard = np.round(overall_metrics_table.loc["MARD", "icgmSensorResults"], 2)
    mbe = np.round(overall_metrics_table.loc["MBE", "icgmSensorResults"], 2)

    title = (
        "{} sensors generated that meet {}% of iCGM Special Controls:".format(
            n_sensors, percent_pass
        )
        + "<br>Criterion A-G = {} ".format(
            SPECIAL_CONTROLS_CRITERIA_THRESHOLDS)
        + "<br>NOISE={}, ".format(np.round(noise_coefficient, 1))
        + "BIAS Type={}, ".format(bias_type)
        + "Drift={}, Drift Range={}, Oscillations={}, ".format(
            bias_drift_type,
            np.round(bias_drift_range, 2),
            np.round(bias_drift_oscillations, 4)
        )
        + "<br>Aggregate Sensor Performance: "
        + "MARD={}, MBE={}, SNR={}, Loss Score={}".format(
            mard, mbe, snr_db, loss_score
        )
    )

    true_trace = go.Scattergl(
        name="true bg",
        x=np.arange(0, len(true_bg_trace)*5, 5),
        y=true_bg_trace,
        hoverinfo="y+name+x",
        mode='markers+lines',
        marker=dict(
            size=2,
            color="blue"
        )
    )

    bg_axis = dict(
        tickvals=[0, 40, 54, 70, 140, 180, 250, 400],
        fixedrange=True,
        hoverformat=".0f",
        zeroline=False,
        showgrid=True,
        gridcolor="#c0c0c0",
        title=dict(
            text="Blood Glucose<br>(mg/dL)",
            font=dict(
                size=12
            )
        )
    )

    layout = go.Layout(
        title=title,
        showlegend=True,
        plot_bgcolor="white",
        yaxis=bg_axis,
        dragmode="pan",
        hovermode="x"
    )

    fig = go.Figure(data=[true_trace], layout=layout)

    for n in range(0, n_sensors):
        fig.add_trace(go.Scattergl(
            name="icgm_{}".format(n),
            x=np.arange(0, len(true_bg_trace)*5, 5),
            y=icgm_traces[n, :],
            hoverinfo="skip",
            mode='lines',
            opacity=0.75,
            line=dict(
                color="orange",
                width=1
            )
        ))

    plot(fig)

    return
