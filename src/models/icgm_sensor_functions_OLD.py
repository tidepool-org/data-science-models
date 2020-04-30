#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:03:31 2019
@author: ed nykaza, jason meno

"""

# %% REQUIRED LIBRARIES
import sys
import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import johnsonsu
from scipy.optimize import curve_fit
import datetime
from pyloopkit.dose import DoseType

# %% FUNCTIONS, CLASSES, AND CONSTANTS

# CONSTANTS
EPS = sys.float_info.epsilon
MICRO = 1E-6


# FUNCTIONS
def create_dataset(
        kind="linear",
        N=288*10,
        min_value=40,
        max_value=400,
        time_interval=5,
        flat_value=np.nan,
        oscillations=1,
        random_seed=89
):

    np.random.seed(random_seed)
    df = pd.DataFrame(np.arange(0, N * 5, 5), columns=["time"])

    if "flat" in kind:
        if pd.isnull(flat_value):
            flat_value = np.random.randint(min_value, max_value + 1, 1)
        df["value"] = np.ones(N) * flat_value

    elif "linear" in kind:
        df["value"] = np.round(np.linspace(min_value, max_value, N))

    elif "sine" in kind:
        df["value"] = np.round(
            np.sin(np.linspace(0, 2 * np.pi * oscillations, N))
            * ((max_value - min_value) / 2)
            + ((max_value + min_value) / 2)
        )

    elif kind == "random":
        df["value"] = np.random.randint(min_value, max_value + 1, N)

    df["rate"] = (df["value"] - df["value"].shift(1)) / 5
    df["rate"].fillna(0, inplace=True)

    input_names = [
        "TRUE.kind",
        "TRUE.N",
        "TRUE.min_value",
        "TRUE.max_value",
        "TRUE.time_interval",
        "TRUE.flat_value",
        "TRUE.oscillations",
        "TRUE.random_seed",
    ]

    input_df = pd.DataFrame(
        [
            kind,
            N,
            min_value,
            max_value,
            time_interval,
            flat_value,
            oscillations,
            random_seed,
        ],
        columns=["icgmSensorResults"],
        index=input_names
    )

    return df, input_df


def snr(cgm_ts):
    cgm_signal = cgm_ts.rolling(window=3, center=True).mean()
    cgm_noise = cgm_ts - cgm_signal
    rms_cgm_signal = np.sqrt(np.mean(cgm_signal**2))
    rms_cgm_noise = np.sqrt(np.mean(cgm_noise**2))
    snr_cgm = (rms_cgm_signal / rms_cgm_noise) ** 2
    snr_cgm_dB = 10 * np.log10(snr_cgm)

    return snr_cgm_dB


def lower_onesided_95p_CB_binomial(number_success, total_trials):
    ''' calculate the lower one-sided 95% confidence bound
    for a binomail distribution
    '''
    Ns = number_success
    N = total_trials
    Nf = N - Ns

    if N > 0:
        LB_95 = (Ns / N) - ((1.644854 / N) * sqrt(Ns * Nf / N))
    else:
        LB_95 = np.nan

    return LB_95


def find_johnson_params(x, a, b, mu, sigma):
    '''
    ppf is inverse of cdf
        input x: the percent of values (0, 1)
        output: gives the value (bias error) at the cutpoint x
    '''
    return johnsonsu.ppf(x, a=a, b=b, loc=mu, scale=sigma)


def get_95percent_bounds(percent_values_within):
    # percent of icgm values that should meet each criterion
    lower_bound = (1 - percent_values_within) / 2
    upper_bound = lower_bound + percent_values_within

    return lower_bound, upper_bound


def generate_icgm_sensors(
        true_bg_trace,
        dist_params,  # [a, b, mu, sigma]
        n_sensors=100,  # (suggest 100 for speed, 1000 for thoroughness)
        bias_type="percentage_of_value",  # (constant_offset, percentage_of_value)
        bias_drift_type="none",  # options (none, linear, random)
        bias_drift_range=[0.95, 1.05],  # (suggest keeping within +/-15%)
        bias_drift_oscillations=0,  # opt for random drift (max of 2)
        noise_coefficient=0,  # (0 ~ 60dB, 5 ~ 36 dB, 10, 30 dB)
        delay=5,  # (suggest 0, 5, 10, 15)
        random_seed=0,
):
    # set a random seed for reproducibility

    np.random.seed(seed=random_seed)
    true_matrix = np.tile(true_bg_trace, (n_sensors, 1))

    # get the initial bias
    a, b, mu, sigma = dist_params
    initial_bias = johnsonsu.rvs(a=a, b=b, loc=mu, scale=sigma, size=n_sensors)

    # add noise
    noise = np.random.normal(
        loc=0,
        scale=np.max([noise_coefficient, EPS]),
        size=(n_sensors, len(true_bg_trace))
    )

    # bias drift
    if "none" in bias_drift_type:
        drift_multiplier = np.ones(np.shape(true_matrix))
        phi = 0

    if "linear" in bias_drift_type:
        drift_multiplier = (
            np.linspace(
                bias_drift_range[0],
                bias_drift_range[1],
                len(true_bg_trace)
            )
        )
        drift_multiplier = np.tile(drift_multiplier, (n_sensors, 1))

    if "random" in bias_drift_type:
        phi = np.random.uniform(low=-np.pi, high=np.pi, size=n_sensors)

        if bias_drift_oscillations == 0:
            bias_drift_oscillations = 1/32
        t = np.linspace(
            0,
            (bias_drift_oscillations * np.pi),
            len(true_bg_trace)
        )
        t_matrix = np.tile(t, (n_sensors, 1))
        phi_matrix = np.tile(phi, (len(true_bg_trace), 1)).T
        sn = np.sin(t_matrix + phi_matrix)

        drift_multiplier = (
            np.interp(sn, (-1, 1), (bias_drift_range[0], bias_drift_range[1]))
        )

    # if the bias type is percentage_of_value (varies by value)
    if "percentage_of_value" in bias_type:
        # the bias factor must be positive, so normalize
        # by the lowest possible bias value ~ -55 given bounds
        # of -50 to 50 for the johnson su distribution
        # TODO: in next version make this a parameter solved by optimization
        # with a range of 50 to 150.
        norm_factor = 55
    else:
        norm_factor = 0

    bias_factor = (norm_factor + initial_bias) / (np.max([norm_factor, 1]))
    bias_factor_matrix = np.tile(bias_factor, (len(true_bg_trace), 1)).T
    iCGM = ((true_matrix * bias_factor_matrix) * drift_multiplier) + noise
    # else:
    #     bias_matrix = np.tile(initial_bias, (len(true_bg_trace), 1)).T
    #     iCGM = ((true_matrix + bias_matrix) * drift_multiplier) + noise

    # add delay or lag to the iCGM traces
    delay_steps = np.int(np.round(delay / 5))
    delayed_iCGM = np.insert(
        values=iCGM[:, 0:1],
        obj=np.zeros(delay_steps, dtype=int),
        arr=iCGM[:, :-delay_steps],
        axis=1
    )

    # capture the individual sensor characertistics for future simulation
    ind_sensor_properties = pd.DataFrame(index=[np.arange(0, n_sensors)])
    ind_sensor_properties["initial_bias"] = initial_bias
    ind_sensor_properties["phi_drift"] = phi

    # also capture the global sensor parameters (for two reasons)
    # 1. so that all of the parameters to simulate iCGM are in one location
    # 2. for future versions of iCGM sensor simulator that allows individual
    # sensors to have variable, noise, bias_drift_oscillations, delay, etc.
    ind_sensor_properties["bias_drift_range_start"] = bias_drift_range[0]
    ind_sensor_properties["bias_drift_range_end"] = bias_drift_range[1]
    ind_sensor_properties["bias_drift_oscillations"] = bias_drift_oscillations
    ind_sensor_properties["bias_norm_factor"] = norm_factor
    ind_sensor_properties["noise_coefficient"] = noise_coefficient
    ind_sensor_properties["delay"] = delay
    ind_sensor_properties["random_seed"] = random_seed

    return delayed_iCGM, ind_sensor_properties


def get_icgm_value(
    true_bg_value,
    at_time=0,
    random_seed=0,
    initial_bias=0,
    phi_drift=0,
    bias_drift_range=[0.95, 1.05],
    bias_drift_oscillations=0,
    bias_norm_factor=55,
    noise_coefficient=2.5,
    bias_drift_type="random" # ("random", "none", "linear")
):
    '''
    This function retrns an iCGM value given a true bg value at time (t),
    and the icgm sensor characteristics. Please note that this function
    does not take into account time delay. If there is time delay
    (e.g., 10 minutes), pass in the true value at time t - 10.

    Parameters
    ----------
    true_bg_value : float
        mg/dL.
    at_time : int, optional
        These are time indices 0, 1, ... T, where each index is 5 minutes.
        The default is 0.
    random_seed : int, optional
        For reproducibility.
        The default is 0.
    initial_bias : float, optional
        Initial Bias of individual iCGM Sensor.
        The default is 0.
    phi_drift : TYPE, optional
        Phase of Bias Dirft for individual iCGM Sensor.
        The default is 0.
    bias_drift_range : TYPE, optional
        Bias drift range of all iCGM Sensors in batch.
        The default is [0.95, 1.05].
    bias_drift_oscillations : TYPE, optional
        Bias drift oscillations of all iCGM Sensors in batch.
        The default is 0.
    bias_norm_factor : TYPE, optional
        Bias normalization factor of all iCGM Sensors in batch.
        The default is 55.
    noise_coefficient : TYPE, optional
        Noise coefficient (sigma of mean-centered white guassian noise),
        used for all iCGM Sensors in batch.
        The default is 2.5.

    Returns
    -------
    iCGM : float
        iCGM value at time (t).

    '''

    # random seed for reproducibility
    np.random.seed(seed=random_seed)

    # noise component
    noise = np.random.normal(
        loc=0,
        scale=np.max([noise_coefficient, EPS]),
        size=288*10
    )

    # bias of individual sensor
    bias_factor = (
        (bias_norm_factor + initial_bias) / (np.max([bias_norm_factor, 1]))
    )

    if bias_drift_type == "random":

        # bias drift component over 10 days with cgm point every 5 minutes
        t = np.linspace(
            0,
            (bias_drift_oscillations * np.pi),
            288 * 10  # this is the number of cgm points in 11 days
        )
        sn = np.sin(t + phi_drift)

        drift_multiplier = (
            np.interp(sn, (-1, 1), (bias_drift_range[0], bias_drift_range[1]))
        )

    if bias_drift_type == "linear":
        print("NO LINEAR TYPE IN ICGM VALUE GENERATOR - NEEDS TO BE WRITTEN")

    if bias_drift_type == "none":
        drift_multiplier = np.ones(288*10)

    iCGM = (
            ((true_bg_value * bias_factor) * drift_multiplier[at_time])
            + noise[at_time]
    )

    return iCGM, bias_factor, noise, drift_multiplier


def coefficient_of_variation(value):
    N = len(value)

    if N > 0:
        cv = np.std(value) / np.mean(value)
    else:
        cv = np.nan

    return cv


def upper_onesided_95p_CB_norm_dist(value):
    ''' calculate the upper one-sided 95% confidence bound
    assuming a normal distribution
    '''
    N = len(value)

    if N > 0:
        UB_95 = np.mean(value) + (1.644854 * (np.std(value) / sqrt(N)))
    else:
        UB_95 = np.nan

    return UB_95


def johnsonsu_icgm_sensor(
        x,  # [a, b , mu, sigma, noise_coefficient, bias_drift_range_min, bias_drift_range_max, bias_drift_oscillations]
        true_bg_trace,
        icgm_special_controls=[0.85, 0.70, 0.80, 0.98, 0.99, 0.99, 0.87],
        n_sensors=100,
        bias_type="constant_offset",  # (constant_offset, percentage_of_value)
        bias_drift_type="none",  # options (none, linear, random)
        delay=5,
        random_seed=0,
        verbose=False,
        use_g6_criteria=False,
):

    # skip distributions that are unrealistic
    dist_min = johnsonsu.ppf(0.0001, a=x[0], b=x[1], loc=x[2], scale=x[3])
    dist_max = johnsonsu.ppf(0.9999, a=x[0], b=x[1], loc=x[2], scale=x[3])

    dist_range = np.nan
    if not np.isinf(dist_min):
        if not np.isinf(dist_max):
            dist_range = dist_max - dist_min

    if (
        (np.abs(dist_min) > 15)
        | (np.abs(dist_max) < 10)
        | (np.abs(dist_max) > 100)
        | (dist_range < 5)
        | (dist_range > 100)
        | (pd.isnull(dist_range))
    ):
        loss = 10000
    else:

        icgm_traces, _ = generate_icgm_sensors(
                true_bg_trace,
                dist_params=x[:4],
                n_sensors=n_sensors,
                bias_type=bias_type,
                bias_drift_type=bias_drift_type,
                bias_drift_range=x[5:7],
                bias_drift_oscillations=x[7],
                noise_coefficient=x[4],
                delay=delay,
                random_seed=random_seed,
        )

        df = preprocess_data(
                true_bg_trace,
                icgm_traces,
                icgm_range=[40, 400],
                ysi_range=[0, 900]
        )

        ''' icgm special controls '''
        acc_results = calc_icgm_sc_table(df, "generic")

        ''' new loss function '''
        if use_g6_criteria:
            g6_loss, g6_table = calc_dexcom_loss(df, n_sensors)
        else:
            g6_loss, g6_table = np.nan, np.nan

        loss, percent_pass = (
            calc_icgm_special_controls_loss(acc_results, g6_loss)
        )

        if verbose:
            print("johnsonsu paramters: {}".format(x))
            print("loss=", loss)
            print("percent pass=", percent_pass)
            print("accuray results=", acc_results)
            print("g6 results=", g6_table)
        # else:
            # print(".", end=" ")

    return loss


def define_bins(values, bin_values, bin_names):
    ''' here is an example:
    values = bg_df["icgm"].values
    bin_values = np.array([39.5, 70, 180.5, 400.5])
    bin_names = ("[40, 70)", "[70, 180]", "(180, 400]")
    '''
    return pd.cut(values, bin_values, labels=bin_names)


# preprocess the icgm and true bg data
def preprocess_data(
        true_array,
        icgm_matrix,
        icgm_range=[40, 400],
        ysi_range=[0, 900]
):
    ''' preprocess data to streamline metric calculations '''

    # default icgm and ysi ranges [40, 400] and [0, 900]
    icgm_min, icgm_max = icgm_range
    ysi_min, ysi_max = ysi_range

    # create a bg dataframe
    n_sensors = np.shape(icgm_matrix)[0]
    bg_df = pd.DataFrame(np.tile(true_array, n_sensors), columns=["ysi"])
    bg_df["icgm"] = icgm_matrix.reshape((-1, 1))

    # calculate the rates
    ysi_rate_array = (true_array - np.roll(true_array, 1)) / 5
    ysi_rate_array[0] = 0
    bg_df["ysiRate"] = np.tile(ysi_rate_array, n_sensors)

    icgm_rate_matrix = (icgm_matrix - np.roll(icgm_matrix, 1, axis=1)) / 5
    icgm_rate_matrix[:, 0] = 0
    bg_df["icgmRate"] = icgm_rate_matrix.reshape((-1, 1))

    # calcualte the icgm error (difference and percentage)
    icgm_values = bg_df["icgm"].values
    ysi_values = bg_df["ysi"].values
    icgm_error = icgm_values - ysi_values
    bg_df["icgmError"] = icgm_error
    abs_difference_error = np.abs(icgm_error)
    bg_df["absError"] = abs_difference_error
    bg_df["absRelDiff"] = 100 * abs_difference_error / ysi_values
    abs_percent_error = np.abs((icgm_values / ysi_values) - 1)
    bg_df["absErrorPercent"] = abs_percent_error

    ''' precalculate bg value bins '''
    # measurement range
    # subtract/add 0.5 to account for rounding (e.g., 39.5 = 40)
    bg_df["withinMeasRange"] = (
        (icgm_values >= icgm_min) & (icgm_values < icgm_max)
    )

    # less than 70 and greater than 180
    bg_df["icgm < 70"] = icgm_values < 70
    bg_df["icgm > 180"] = icgm_values > 180
    bg_df["ysi < 70"] = ysi_values < 70
    bg_df["ysi > 180"] = ysi_values > 180

    # icgm special control bins
    bin_values = np.array(
        [ysi_min, icgm_min - MICRO, 70 - MICRO, 180, icgm_max, ysi_max]
    )

    bin_names = (
        "({}, {})".format(ysi_min, icgm_min),
        "[{}, 70)".format(icgm_min),
        "[70, 180]",
        "(180, {}]".format(icgm_max),
        "({}, {})".format(icgm_max, ysi_max)
    )
    bg_df["icgmBins"] = define_bins(icgm_values, bin_values, bin_names)

    # icgm special control secondary bins
    bin_values = np.array(
        [icgm_min - MICRO, 54 - MICRO, 70 - MICRO, 180, 250, icgm_max]
    )
    bin_names = (
        "[{}, 54)".format(icgm_min),
        "[54, 70)",
        "[70, 180]",
        "(180, 250]",
        "(250, {}]".format(icgm_max),
    )

    bg_df["icgmBins2"] = define_bins(icgm_values, bin_values, bin_names)

    # ysi secondary bins
    bg_df["ysiBins2"] = define_bins(ysi_values, bin_values, bin_names)

    # icgm concurrence bins
    bin_values = np.array(
        [ysi_min,
         icgm_min - MICRO,
         60.499, 80.499, 120.499, 160.499, 200.499, 250.499, 300.499, 350.499,
         icgm_max,
         ysi_max]
    )
    bin_names = (
        "({}, 40)".format(ysi_min),
        "[40, 60]",
        "[61, 80]",
        "[81, 120]",
        "[121, 160]",
        "[161, 200]",
        "[201, 250]",
        "[251, 300]",
        "[301, 350]",
        "[351, 400]",
        "(400, {})".format(ysi_max)
    )
    bg_df["icgmConcBins"] = define_bins(icgm_values, bin_values, bin_names)

    # ysi concurrence bins
    bg_df["ysiConcBins"] = define_bins(ysi_values, bin_values, bin_names)

    # icgm difference and percentage bins
    for v in [15, 20, 40]:
        bg_df["within+/-{}mg/dL".format(v)] = abs_difference_error < v
        bg_df["within+/-{}%".format(v)] = abs_percent_error < (v/100)

    bg_df["gte40mg/dL"] = abs_difference_error >= 40
    bg_df["gte40%"] = abs_percent_error >= 0.40

    ''' precalculate bg rate bins '''
    icgm_rates = bg_df["icgmRate"].values
    ysi_rates = bg_df["ysiRate"].values

    # less than < -2, < -1, and greater than > 1, > 2
    bg_df["icgmRate < -1"] = icgm_rates < -1
    bg_df["icgmRate > 1"] = icgm_rates > 1
    bg_df["ysiRate < -2"] = ysi_rates < -2
    bg_df["ysiRate > 2"] = ysi_rates > 2

    # icgm rate bins
    bin_values = np.array([-100, -2.0001, -1.0001, 0 - EPS, 1, 2, 100])
    bin_names = ["<-2", "[-2, -1)", "[-1, 0)", "[0, 1]", "(1, 2]", ">2"]
    bg_df["icgmRateBins"] = define_bins(icgm_rates, bin_values, bin_names)

    # ysi rate bins
    bg_df["ysiRateBins"] = define_bins(ysi_rates, bin_values, bin_names)

    ''' calculate time bins '''
    # calculate the day of the sensor
    sensor_day = np.floor((np.arange(0, len(true_array)) / 288) + 1)
    day_matrix = np.tile(sensor_day, (np.shape(icgm_matrix)[0], 1))
    bg_df["day"] = day_matrix.reshape((-1, 1))

    # NOTE: dexcom defines beginning as days 1-2, middle as 4-5, and
    # end as 7 and/or 10. We'll define 1-3, 4-6, and 7-10
    bin_values = np.array([0, 3, 6, 10])
    bin_names = ("Beginning", "Middle", "End")

    bg_df["wearPeriod"] = (
        define_bins(bg_df["day"].values, bin_values, bin_names)
    )

    return bg_df


def calc_percent_within_mgdL(df, within_threshold):
    bt_40_70 = ((df["withinMeasRange"]) & (df["icgm < 70"]))
    within_mgdL = (
        bt_40_70 & (df["within+/-{}mg/dL".format(within_threshold)])
    )
    n_within_mgdL = within_mgdL.sum()
    total_within_mgdL = bt_40_70.sum()
    percent_within_mgdL = 100 * n_within_mgdL / total_within_mgdL

    return percent_within_mgdL, n_within_mgdL, total_within_mgdL


def calc_percent_within_percent(df, within_threshold, icgm_range="70-400"):
    if "70-400" in icgm_range:
        i_range = ((df["withinMeasRange"]) & (~df["icgm < 70"]))
    else:  # This is the case for criterion G (across entire measurement range)
        i_range = df["withinMeasRange"]

    within_percent = (
        i_range & (df["within+/-{}%".format(within_threshold)])
    )

    n_within_percent = within_percent.sum()
    total_within_percent = i_range.sum()
    percent_within_percent = 100 * n_within_percent / total_within_percent

    return percent_within_percent, n_within_percent, total_within_percent


def calc_percent_within(df, within_threshold):

    _, n_within_mgdL, total_within_mgdL = (
        calc_percent_within_mgdL(df, within_threshold)
    )

    _, n_within_percent, total_within_percent = (
        calc_percent_within_percent(df, within_threshold)
    )

    n_meet_criterion = n_within_mgdL + n_within_percent
    total_all = total_within_mgdL + total_within_percent

    percent_within = 100 * n_meet_criterion / total_all
    percent_within_95_lower_bound = lower_onesided_95p_CB_binomial(
        n_meet_criterion,
        total_all
    ) * 100

    return percent_within, percent_within_95_lower_bound


def calc_mbe(df):
    return np.mean(df.loc[df["withinMeasRange"], "icgmError"])


def calc_mard(df):
    ''' Mean Absolute Relative Deviation (MARD)
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5375072/
    '''

    abs_relative_difference_in_measurement_range = (
        df.loc[df["withinMeasRange"], "absRelDiff"]
    )
    return np.mean(abs_relative_difference_in_measurement_range)


def calc_icgm_sc_table(df, sensor="generic"):
    ''' iCGM special controls Table '''
    if "generic" in sensor:
        col_name = "icgmSpecialControls"
        icgm_accuracy_thresholds = (
            [85, 70, 80, 98, 99, 99, 87, 100, 100, 99, 99]
        )
    elif "g6" in sensor:
        # NOTE: this is combining adult and peds data submitted with
        # icgm de novo submission
        # https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN170088.pdf
        col_name = "dexG6"
        a_thresh = lower_onesided_95p_CB_binomial(
            ((0.885 * 1920) + (0.761 * 352)),
            (1920 + 352)
        ) * 100

        d_thresh = lower_onesided_95p_CB_binomial(
            ((0.993 * 1920) + (0.938 * 352)),
            (1920 + 352)
        ) * 100

        b_thresh = lower_onesided_95p_CB_binomial(
            ((0.741 * 9543) + (0.800 * 3142)),
            (9543 + 3142)
        ) * 100

        e_thresh = lower_onesided_95p_CB_binomial(
            ((0.993 * 9543) + (0.995 * 3142)),
            (9543 + 3142)
        ) * 100

        c_thresh = lower_onesided_95p_CB_binomial(
            ((0.855 * 7956) + (0.858 * 2276)),
            (7956 + 2276)
        ) * 100

        f_thresh = lower_onesided_95p_CB_binomial(
            ((0.999 * 7956) + (0.999 * 2276)),
            (7956 + 2276)
        ) * 100

        # NOTE: this is in the trend table
        j_thresh = 100 - ((
            ((0.000 * 1734) + (0.001 * 1376) + (0.000 * 546) + (0.000 * 423))
            / (1734 + 1376 + 546 + 423)
        ) * 100)

        k_thresh = 100 - ((
            ((0.002 * 463) + (0.000 * 2077) + (0.005 * 211) + (0.001 * 686))
            / (463 + 2077 + 211 + 686)
        ) * 100)

        icgm_accuracy_thresholds = ([
            a_thresh,
            b_thresh,
            c_thresh,
            d_thresh,
            e_thresh,
            f_thresh,
            88.7,
            100,
            100,
            j_thresh,
            k_thresh
        ])

    else:
        sys.exit("Error: only 'generic' and dex 'g6' are defined at this time")

    icgm_sc_index = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

    icgm_sc_table = pd.DataFrame(
        icgm_accuracy_thresholds,
        columns=[col_name],
        index=icgm_sc_index
    )

    icgm_sc_table["nPairs"] = np.nan
    icgm_sc_table["icgmSensorResults"] = np.nan

    # Criterion A through F
    icgm_within = [15, 15, 15, 40, 40, 40]
    icgm_criterion = ["A", "B", "C", "D", "E", "F"]
    icgm_range = [
        "[40, 70)",
        "[70, 180]",
        "(180, 400]",
        "[40, 70)",
        "[70, 180]",
        "(180, 400]",
    ]

    for w, c, r in zip(icgm_within, icgm_criterion, icgm_range):
        if "[40, 70)" in r:
            _, n_meet_criterion, n_pairs = (
                calc_percent_within_mgdL(df[df["icgmBins"] == r], w)
            )

        else:
            _, n_meet_criterion, n_pairs = (
                calc_percent_within_percent(df[df["icgmBins"] == r], w)
            )

        percent_within_95_lower_bound = lower_onesided_95p_CB_binomial(
            n_meet_criterion,
            n_pairs
        ) * 100

        icgm_sc_table.loc[c, ["nPairs", "icgmSensorResults"]] = (
            n_pairs, percent_within_95_lower_bound
        )

    # Criterion G
    _, n_meet_criterion, n_pairs = (
        calc_percent_within_percent(df, 20, icgm_range="G")
    )

    percent_within_95_lower_bound = lower_onesided_95p_CB_binomial(
        n_meet_criterion,
        n_pairs
    ) * 100

    icgm_sc_table.loc["G", ["nPairs", "icgmSensorResults"]] = (
        n_pairs, percent_within_95_lower_bound
    )

    # Criterion H
    total_icgm_lt70 = df["icgm < 70"].sum()
    criterion_H = 100 - ((
        np.sum((df["icgm < 70"]) & (df["ysi > 180"])) / total_icgm_lt70
    ) * 100)
    icgm_sc_table.loc["H", ["nPairs", "icgmSensorResults"]] = (
        total_icgm_lt70, criterion_H
    )

    # Criterion I
    total_icgm_gt180 = df["icgm > 180"].sum()
    criterion_I = 100 - ((
        np.sum((df["icgm > 180"]) & (df["ysi < 70"])) / total_icgm_gt180
    ) * 100)
    icgm_sc_table.loc["I", ["nPairs", "icgmSensorResults"]] = (
        total_icgm_gt180, criterion_I
    )

    # Criterion J
    ''' Criterion J: There shall be no more than 1% of iCGM measurements
    that indicate a positive glucose rate of change > 1 mg/dL/min when the
    corresponding true negative glucose rate of change is < -2 mg/dL/min
    as determined by the corresponding blood glucose measurements.'''

    total_icgm_rate_gt1 = df["icgmRate > 1"].sum()
    criterion_J = 100 - ((
        np.sum((df["icgmRate > 1"]) & (df["ysiRate < -2"]))
        / total_icgm_rate_gt1
    ) * 100)

    icgm_sc_table.loc["J", ["nPairs", "icgmSensorResults"]] = (
        total_icgm_rate_gt1, criterion_J
    )

    # Criterion K
    ''' Criterion K: There shall be no more than 1% of iCGM measurements
    that indicate a negative glucose rate of change <  -1 mg/dL/min when the
    corresponding true positive glucose rate of change is > 2 mg/dL/min
    as determined by the corresponding blood glucose measurements.'''
    total_icgm_rate_lt_neg1 = df["icgmRate < -1"].sum()
    criterion_K = 100 - ((
        np.sum((df["icgmRate < -1"]) & (df["ysiRate > 2"]))
        / total_icgm_rate_lt_neg1
    ) * 100)

    icgm_sc_table.loc["K", ["nPairs", "icgmSensorResults"]] = (
        total_icgm_rate_lt_neg1, criterion_K
    )

    return icgm_sc_table


def capture_settings(
    n_sensors,
    use_g6_accuracy_in_loss,
    bias_type,
    bias_drift_type,
    delay,
    random_seed
):
    settings_list = [
        "n_sensors",  "use_g6_accuracy_in_loss",
        "bias_type", "bias_drift_type", "delay", "random_seed",
    ]

    settings_table = pd.DataFrame(
        columns=["icgmSensorResults"],
        index=settings_list
    )

    settings_values = [
        n_sensors, use_g6_accuracy_in_loss,
        bias_type, bias_drift_type, delay, random_seed
    ]
    settings_table.loc[settings_list, "icgmSensorResults"] = settings_values

    return settings_table


def calc_overall_metrics(df):
    overall_table = pd.DataFrame(columns=["icgmSensorResults"])
    ''' Mean Absolute Relative Deviation (MARD)
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5375072/
    '''
    overall_table.loc["MARD", "icgmSensorResults"] = calc_mard(df)

    ''' mean bias error '''
    overall_table.loc["MBE", "icgmSensorResults"] = calc_mbe(df)

    ''' mean bias error 95% upper bound '''
    mbe_95p_upper_bound = (
        upper_onesided_95p_CB_norm_dist(df["icgmError"].values)
    )
    overall_table.loc["MBE_UB95", "icgmSensorResults"] = mbe_95p_upper_bound

    # coefficient of variation
    cov = coefficient_of_variation(df["icgmError"].values)
    overall_table.loc["CV", "icgmSensorResults"] = cov

    ''' overall smoothness (signal-to-noise) '''
    overall_table.loc["SNR", "icgmSensorResults"] = snr(df["icgm"])

    return overall_table


def calc_g6_table1A(df, n_sensors):
    ''' Dexcom G6 Table 1A '''
    dex_g6_table_1A_data = np.array([324, 25101, 91.7, 90.6, 87.8, 9.8])
    dex_g6_table_1A_cols = [
        "nSubjectsOrSensors",
        "nPairs",
        "percentWithin20/20%YSI",
        "95%LB_percentWithin20/20%YSI",
        "day1percentWithin20/20%YSI",
        "MARD%"
    ]
    dex_g6_table_1A = pd.DataFrame(
        np.reshape(dex_g6_table_1A_data, (1, -1)),
        columns=dex_g6_table_1A_cols,
        index=["dexG6"]
    )

    table_1A = dex_g6_table_1A.T

    # calculate data that corresponds with table 1A
    n_pairs = df["withinMeasRange"].sum()

    perc_within_20_20p, percent_within_20_20p_95LB = (
        calc_percent_within(df, 20)
    )

    day1_perc_within_20_20p, _ = calc_percent_within(df[df["day"] == 1], 20)

    mard = calc_mard(df)

    table_1A["icgmSensorResults"] = [
        n_sensors,
        n_pairs,
        perc_within_20_20p,
        percent_within_20_20p_95LB,
        day1_perc_within_20_20p,
        mard
    ]
    return table_1A


def calc_g6_table1BF(df, n_sensors, table_letter="B"):
    ''' Dexcom G6 Table 1B & 1F '''

    if "B" in table_letter:
        glucose_range = "icgmBins2"

    # Table 1B
        dex_g6_table_1BF_data = np.array([
            [159, 383, 84.3, 90.6, 98.4, np.nan, np.nan, np.nan, -6.9, 13.8],
            [159, 1537, 89.6, 95.1, 99.5, np.nan, np.nan, np.nan, -0.5, 11.5],
            [159, 9453, np.nan, np.nan, np.nan, 73.9, 86.6, 99.3, -2.8, 10.9],
            [159, 4093, np.nan, np.nan, np.nan, 80.2, 92.1, 99.9, -10.0, 9.3],
            [159, 3863, np.nan, np.nan, np.nan, 91.1, 97.7, 100.0, -3.8, 7.1],
        ])

    elif "F" in table_letter:
        glucose_range = "ysiBins2"

        # Table 1F
        dex_g6_table_1BF_data = np.array([
            [159, 483, 88.2, 95.9, 99.8, np.nan, np.nan, np.nan, 6.0, 15.8],
            [159, 1783, 88.8, 96.1, 99.9, np.nan, np.nan, np.nan, 4.0, 12.4],
            [159, 8713, np.nan, np.nan, np.nan, 76.8, 89.0, 99.6, -0.8, 10.3],
            [159, 3940, np.nan, np.nan, np.nan, 83.0, 92.7, 99.8, -7.2, 8.8],
            [159, 4410, np.nan, np.nan, np.nan, 83.4, 93.3, 99.8, -13.5, 8.6],
        ])

    else:
        sys.exit("Error: only tables 1B and 1F are defined at this time")

    dex_g6_table_1BF_index = [
        "[40, 54)", "[54, 70)", "[70, 180]", "(180, 250]", "(250, 400]"
    ]

    dex_g6_table_1BF_cols = [
        "nSubjectsOrSensors",
        "nPairs",
        "percentWithin15YSI",
        "percentWithin20YSI",
        "percentWithin40YSI",
        "percentWithin15%YSI",
        "percentWithin20%YSI",
        "percentWithin40%YSI",
        "MBE",
        "MARD%"
    ]

    dex_g6_table_1BF = pd.DataFrame(
        dex_g6_table_1BF_data,
        columns=dex_g6_table_1BF_cols,
        index=dex_g6_table_1BF_index
    )

    table_1BF = pd.DataFrame(
        dex_g6_table_1BF.stack(dropna=False),
        columns=["dexG6"]
    )
    table_1BF["icgmSensorResults"] = np.nan

    for measurement_range in dex_g6_table_1BF.index:
        subset = df[df[glucose_range] == measurement_range]

        # calculate number of icgm-ysi pairs
        for v in [15, 20, 40]:
            if "54" in measurement_range:
                perc_within_mgdL, _, n_pairs = (
                    calc_percent_within_mgdL(subset, v)
                )

                table_1BF.loc[
                    (measurement_range, "percentWithin{}YSI".format(v)),
                    "icgmSensorResults"
                ] = perc_within_mgdL

            else:
                perc_within_percent, _, n_pairs = (
                    calc_percent_within_percent(subset, v)
                )

                table_1BF.loc[
                    (measurement_range, "percentWithin{}%YSI".format(v)),
                    "icgmSensorResults"
                ] = perc_within_percent

            table_1BF.loc[
                (measurement_range, 'nPairs'), "icgmSensorResults"
            ] = n_pairs

        table_1BF.loc[
            (measurement_range, 'nSubjectsOrSensors'), "icgmSensorResults"
        ] = n_sensors

        table_1BF.loc[
            (measurement_range, 'MBE'), "icgmSensorResults"
        ] = calc_mbe(subset)

        table_1BF.loc[
            (measurement_range, 'MARD%'), "icgmSensorResults"
        ] = calc_mard(subset)

    return table_1BF


def calc_g6_table3AC(df, n_sensors, table_letter="A"):
    ''' Dexcom G6 Table 3A '''

    if "A" in table_letter:
        glucose_range1_bins = "icgmConcBins"
        glucose_range2_bins = "ysiConcBins"

        # NOTE: this only contains adult data
        # TODO: update to include peds data too
        dex_g6_table_3AC_data = np.array([
            [159, 104, 13.5, 56.7, 24.0, 3.8, 1.9, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [159, 917, 1.2, 67.8, 27.9, 2.7, 0.2, 0.1, np.nan, np.nan, np.nan, np.nan, np.nan],
            [159, 2275, 0.1, 21.3, 61.4, 16.9, 0.3, 0.1, np.nan, np.nan, np.nan, np.nan, np.nan],
            [159, 3782, np.nan, 0.4, 13.6, 70.3, 15.1, 0.6, 0.0, np.nan, np.nan, np.nan, np.nan],
            [159, 3026, np.nan, np.nan, 0.0, 14.2, 64.3, 20.1, 1.3, 0.0, 0.0, np.nan, np.nan],
            [159, 2597, np.nan, np.nan, np.nan, 0.1, 14.5, 56.7, 26.9, 1.5, 0.2, 0.0, np.nan],
            [159, 2869, np.nan, np.nan, np.nan, np.nan, 0.2, 12.1, 59.4, 25.4, 2.9, 0.0, np.nan],
            [159, 2268, np.nan, np.nan, np.nan, np.nan, np.nan, 0.1, 13.7, 59.1, 25.3, 1.9, np.nan],
            [159, 1212, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.2, 22.3, 63.4, 13.7, 0.5],
            [159, 383, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.8, 43.9, 52.5, 2.9],
            [159, 34, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5.9, 76.5, 17.6],
        ])

    elif "C" in table_letter:
        glucose_range1_bins = "ysiConcBins"
        glucose_range2_bins = "icgmConcBins"
        # NOTE: this table is inverted so that the percentages in each
        # row add up to 100%
        # NOTE: this only contains adult data
        # TODO: update to include peds data too
        dex_g6_table_3AC_data = np.array([
            [159, 27, 51.9, 40.7,  7.4,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
            [159, 1180, 5. , 52.7, 41. ,  1.3,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
            [159, 2191, 1.1, 11.7, 63.7, 23.4,  0. ,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
            [159, 3503, 0.1,  0.7, 11. , 75.8, 12.2,  0.1,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
            [159, 2910, 0.1,  0.1,  0.2, 19.7, 66.9, 13. ,  0.2,  np.nan,  np.nan,  np.nan,  np.nan],
            [159, 2457, np.nan,  0. ,  0.1,  1. , 24.8, 59.9, 14.1,  0.1,  np.nan,  np.nan,  np.nan],
            [159, 2755, np.nan,  np.nan,  np.nan,  0. ,  1.4, 25.3, 61.9, 11.3,  0.1,  np.nan,  np.nan],
            [159, 2383, np.nan,  np.nan,  np.nan,  np.nan,  0. ,  1.7, 30.6, 56.2, 11.3,  0.1,  np.nan],
            [159, 1601, np.nan,  np.nan,  np.nan,  np.nan,  0.1,  0.4,  5.1, 35.9, 48. , 10.5,  0.1],
            [159, 437, np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  0.2,  0.2,  9.6, 38. , 46. ,  5.9],
            [159, 23, np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan, 26.1, 47.8, 26.1]
        ])


    else:
        sys.exit("Error: only tables 3A and 3C are defined at this time")

    dex_g6_table_3AC_index = [
        "(0, 40)",
        "[40, 60]",
        "[61, 80]",
        "[81, 120]",
        "[121, 160]",
        "[161, 200]",
        "[201, 250]",
        "[251, 300]",
        "[301, 350]",
        "[351, 400]",
        "(400, 900)",
    ]

    dex_g6_table_3AC_cols = (
        np.append(["nSubjectsOrSensors", "nPairs"], dex_g6_table_3AC_index)
    )

    dex_g6_table_3AC = pd.DataFrame(
        dex_g6_table_3AC_data,
        columns=dex_g6_table_3AC_cols,
        index=dex_g6_table_3AC_index
    )

    table_3AC = pd.DataFrame(
        dex_g6_table_3AC.stack(dropna=False),
        columns=["dexG6"]
    )

    table_3AC["icgmSensorResults"] = np.nan

    for glucose_range1 in dex_g6_table_3AC_index:

        subset = df[df[glucose_range1_bins] == glucose_range1]

        table_3AC.loc[
            (glucose_range1, 'nSubjectsOrSensors'), "icgmSensorResults"
        ] = n_sensors

        n_pairs = len(subset)
        table_3AC.loc[
            (glucose_range1, 'nPairs'), "icgmSensorResults"
        ] = n_pairs

        for glucose_range2 in dex_g6_table_3AC_index:
            n_concurrent = np.sum(subset[glucose_range2_bins] == glucose_range2)
            if n_pairs > 0:
                concurrency_percent = 100 * n_concurrent / n_pairs
            else:
                concurrency_percent = np.nan
            table_3AC.loc[
                (glucose_range1, glucose_range2), "icgmSensorResults"
            ] = concurrency_percent

    return table_3AC


def calc_g6_table4(df, n_sensors):

    dex_g6_table_4A_data = np.array([
        [159, 463, 53.3, 35.0, 9.9, 1.5, 0.0, 0.2],
        [159, 2077, 7.4, 56.9, 32.5, 2.9, 0.3, 0.0],
        [159, 7986, 0.4, 9.5, 76.9, 12.5, 0.6, 0.1],
        [159, 5199, 0.1, 1.0, 26.2, 60.6, 10.6, 1.6],
        [159, 1734, 0.0, 0.4, 3.1, 26.8, 52.9, 16.8],
        [159, 1367, 0.1, 0.1, 0.8, 5.6, 22.1, 71.3]
    ])

    dex_g6_table_4B_data = np.array([
        [165, 211, 47.9, 37.0, 12.8, 1.9, 0.0, 0.5],
        [165, 686, 6.6, 55.5, 33.8, 3.4, 0.6, 0.1],
        [165, 2048, 0.5, 8.9, 73.7, 15.8, 1.0, 0.0],
        [165, 1666, 0.0, 0.8, 25.5, 62.9, 10.0, 0.8],
        [165, 546, 0.0, 0.4, 4.4, 35.9, 48.0, 11.4],
        [165, 423, 0.0, 0.5, 1.7, 7.1, 23.6, 67.1]
    ])

    dex_g6_table_4_data = np.empty(np.shape(dex_g6_table_4A_data))
    dex_g6_table_4_data[:] = np.nan
    dex_g6_table_4_data[:, 0] = (
        dex_g6_table_4A_data[:, 0] + dex_g6_table_4B_data[:, 0]
    )
    total_adult_peds = dex_g6_table_4A_data[:, 1] + dex_g6_table_4B_data[:, 1]
    dex_g6_table_4_data[:, 1] = total_adult_peds

    dex_g6_table_4_totals = np.tile(total_adult_peds, (6, 1)).T

    dex_g6_table_4_data[:, 2:] = (
        (
            (dex_g6_table_4A_data[:, 2:] / 100)
            * np.tile(dex_g6_table_4A_data[:, 1], (6, 1)).T
        )
        + (
            (dex_g6_table_4B_data[:, 2:] / 100)
            * np.tile(dex_g6_table_4B_data[:, 1], (6, 1)).T
        )
    ) / dex_g6_table_4_totals * 100

    dex_g6_table_4_index = [
        "<-2", "[-2, -1)", "[-1, 0)", "[0, 1]", "(1, 2]", ">2"
    ]

    dex_g6_table_4_cols = (
        np.append(["nSubjectsOrSensors", "nPairs"], dex_g6_table_4_index)
    )

    dex_g6_table_4 = pd.DataFrame(
        dex_g6_table_4_data,
        columns=dex_g6_table_4_cols,
        index=dex_g6_table_4_index
    )

    table_4 = pd.DataFrame(
        dex_g6_table_4.stack(dropna=False),
        columns=["dexG6"]
    )

    table_4["icgmSensorResults"] = np.nan

    for icgm_rate in dex_g6_table_4_index:

        n_pairs = (
            (df["icgmRateBins"] == icgm_rate)
            & (df["withinMeasRange"])
        ).sum()

        table_4.loc[
            (icgm_rate, "nPairs"), "icgmSensorResults"
        ] = n_pairs

        table_4.loc[
            (icgm_rate, "nSubjectsOrSensors"), "icgmSensorResults"
        ] = n_sensors

        for ysi_rate in dex_g6_table_4_index:

            concurrence = (
                (df["icgmRateBins"] == icgm_rate)
                & (df["ysiRateBins"] == ysi_rate)
                & (df["withinMeasRange"])
            ).sum()
            concurrence_percent = concurrence / n_pairs * 100
            table_4.loc[
                (icgm_rate, ysi_rate), "icgmSensorResults"
            ] = concurrence_percent

    return table_4


def calc_g6_table6(df, n_sensors):
    ''' Table 6-A. Sensor Stability Relative to YSI (Accuracy over Time)
    within +/- 15/15, 20/20, & 40/40 over beginning (days 1 and 2),
    middle day (days 4 and 5), and end (days 7 and/or 10)
    '''

    dex_g6_table_6A_data = np.array([
        [159, 6696, 10.9, 76.5, 88.0, 99.6],
        [159, 6464, 9.2, 84.3, 94.6, 99.8],
        [159, 6169, 9.6, 82.3, 92.4, 99.8],
    ])

    dex_g6_table_6B_data = np.array([
        [165, 2167, 9.9, 81.2, 92.1, 99.8],
        [165, 1268, 9.1, 83.1, 93.7, 99.8],
        [165, 2337, 9.4, 83.1, 91.1, 98.5],
    ])

    dex_g6_table_6_data = np.empty(np.shape(dex_g6_table_6A_data))
    dex_g6_table_6_data[:] = np.nan
    dex_g6_table_6_data[:, 0] = (
        dex_g6_table_6A_data[:, 0] + dex_g6_table_6B_data[:, 0]
    )
    total_adult_peds = dex_g6_table_6A_data[:, 1] + dex_g6_table_6B_data[:, 1]
    dex_g6_table_6_data[:, 1] = total_adult_peds

    dex_g6_table_6_totals = np.tile(total_adult_peds, (4, 1)).T

    dex_g6_table_6_data[:, 2:] = (
        (
            (dex_g6_table_6A_data[:, 2:] / 100)
            * np.tile(dex_g6_table_6A_data[:, 1], (4, 1)).T
        )
        + (
            (dex_g6_table_6B_data[:, 2:] / 100)
            * np.tile(dex_g6_table_6B_data[:, 1], (4, 1)).T
        )
    ) / dex_g6_table_6_totals * 100

    dex_g6_table_6_index = ["Beginning", "Middle", "End"]

    dex_g6_table_6_cols = [
        "nSubjectsOrSensors",
        "nPairs",
        "MARD%",
        "percentWithin15/15%YSI",
        "percentWithin20/20%YSI",
        "percentWithin40/40%YSI"
    ]

    dex_g6_table_6 = pd.DataFrame(
        dex_g6_table_6_data,
        columns=dex_g6_table_6_cols,
        index=dex_g6_table_6_index
    )

    table_6 = pd.DataFrame(
        dex_g6_table_6.stack(dropna=False),
        columns=["dexG6"]
    )

    table_6["icgmSensorResults"] = np.nan

    table_6.loc[
        (dex_g6_table_6_index, "nSubjectsOrSensors"), "icgmSensorResults"
    ] = n_sensors

    for wear_period in dex_g6_table_6_index:
        subset = df[df["wearPeriod"] == wear_period]
        n_pairs = subset["withinMeasRange"].sum()

        table_6.loc[
            (wear_period, "nPairs"), "icgmSensorResults"
        ] = n_pairs

        table_6.loc[
            (wear_period, "MARD%"), "icgmSensorResults"
        ] = calc_mard(subset)

        for v in [15, 20, 40]:
            percent_within, _ = (
                calc_percent_within(subset, v)
            )

            table_6.loc[
                (wear_period, "percentWithin{}/{}%YSI".format(v, v)),
                "icgmSensorResults"
            ] = percent_within

    return table_6


def calc_dexcom_loss(df, n_sensors):
    gsc = calc_icgm_sc_table(df, "g6")
    g1a = calc_g6_table1A(df, n_sensors)
    g1b = calc_g6_table1BF(df, n_sensors, "B")
    table6 = calc_g6_table6(df, n_sensors)

    g6_table = pd.concat([
        gsc[["dexG6", "icgmSensorResults"]],
        g1a.loc[["95%LB_percentWithin20/20%YSI", "MARD%"], :],
        g1b,
        table6
    ], sort=False)

    g6_table = g6_table[
        ((g6_table["dexG6"] > -100) & (g6_table["dexG6"] <= 100))
    ]

    y_hat = g6_table["icgmSensorResults"].fillna(-100).values
    y = g6_table["dexG6"].values
    ind_diff = y_hat - y
    ind_loss = np.abs(ind_diff)
    total_loss = np.sum(ind_loss)

    return total_loss, g6_table


def calc_icgm_special_controls_loss(icgm_special_controls_table, g6_loss):
    y_hat = icgm_special_controls_table["icgmSensorResults"].fillna(0).values
    y = icgm_special_controls_table["icgmSpecialControls"].values
    ind_diff = y_hat - y
    neg_penalty = (ind_diff < -1) * 1
    ind_loss = np.abs(ind_diff) ** (1 + neg_penalty)
    percent_pass = (
        100 * np.sum(ind_diff >= 0) / (len(icgm_special_controls_table))
    )
    all_pass_penalty = 100 - percent_pass

    if pd.notnull(g6_loss):
        total_loss = all_pass_penalty + g6_loss
    else:
        total_loss = all_pass_penalty + (np.sum(ind_loss) / percent_pass)

    return total_loss, percent_pass


def get_search_range(
    SPECIAL_CONTROLS_CRITERIA=[0.85, 0.70, 0.80, 0.98, 0.99, 0.99, 0.87],
    SEARCH_SPAN=10,
    BIAS_CATEGORY="NOT_SPECIFIED",
    BIAS_MIN=-50,
    BIAS_MAX=50,
    BIAS_DRIFT_MIN=0.85,
    BIAS_DRIFT_MAX=1.15,
    BIAS_DRIFT_STEP=0.15,
    BIAS_DRIFT_OSCILLATION_MIN=0,
    BIAS_DRIFT_OSCILLATION_MAX=3,
    BIAS_DRIFT_OSCILLATION_STEP=1,
    NOISE_MIN=2.5,  # NOTE: CHANGED TO REQUIRE MINIMUM AMOUNT OF NOISE
    NOISE_MAX=20,
    NOISE_STEP=5
):

    # for completley positive bias
    if BIAS_CATEGORY in "COMPLETE_POSITIVE_BIAS":

        perc_points_40_70 = [
            0.0001,
            SPECIAL_CONTROLS_CRITERIA[0],
            SPECIAL_CONTROLS_CRITERIA[6],
            SPECIAL_CONTROLS_CRITERIA[3],
            0.9999
        ]
        icgm_errors_40_70 = [0, 15, 20, 40, BIAS_MAX]

        a_b_mu_sigma_bounds = (
            [-100, EPS, 10,  1],
            [-10,   10,  100, np.inf]
        )

        if pd.isnull(SEARCH_SPAN):
            SEARCH_SPAN = 1

    # for completley positive bias
    elif BIAS_CATEGORY in "COMPLETE_NEGATIVE_BIAS":

        perc_points_40_70 = [
            0.0001,
            SPECIAL_CONTROLS_CRITERIA[0],
            SPECIAL_CONTROLS_CRITERIA[6],
            SPECIAL_CONTROLS_CRITERIA[3],
            0.9999
        ]
        icgm_errors_40_70 = [0, -15, -20, -40, BIAS_MIN]

        a_b_mu_sigma_bounds = (
            [10, EPS, -100,  1],
            [100,   10,  -10, np.inf]
        )

        if pd.isnull(SEARCH_SPAN):
            SEARCH_SPAN = 1

    else:
        A_LB, A_UB = get_95percent_bounds(SPECIAL_CONTROLS_CRITERIA[0])
        D_LB, D_UB = get_95percent_bounds(SPECIAL_CONTROLS_CRITERIA[3])
        G_LB, G_UB = get_95percent_bounds(SPECIAL_CONTROLS_CRITERIA[6])
        perc_points_40_70 = [
            0.0001, D_LB, G_LB, A_LB, 0.5, A_UB, G_LB, D_UB, 0.9999
        ]
        icgm_errors_40_70 = [
            BIAS_MIN, -40, -20, -15, 0, 15, 20, 40, BIAS_MAX
        ]

        # for no bias
        if BIAS_CATEGORY in "NO_BIAS":

            a_b_mu_sigma_bounds = (
                [0, EPS, 0, 1],
                [EPS, 10, EPS, np.inf]
            )

            if pd.isnull(SEARCH_SPAN):
                SEARCH_SPAN = EPS

        # if no bias is specified
        else:
            a_b_mu_sigma_bounds = (
                [-20, EPS, -20, 5],
                [20, 100, 20, np.inf]
            )

            if pd.isnull(SEARCH_SPAN):
                SEARCH_SPAN = 5

    # get distribution settings that are in the ballpark
    (a_init, b_init, mu_init, sigma_init), pcov = curve_fit(
        find_johnson_params,
        perc_points_40_70,
        icgm_errors_40_70,
        bounds=a_b_mu_sigma_bounds
    )

    # set the search grid ranges
    rranges = (
        slice(
            a_init - SEARCH_SPAN,
            a_init + (SEARCH_SPAN * 2),
            SEARCH_SPAN
        ),
        slice(
            max([b_init - SEARCH_SPAN, EPS]),
            b_init + (SEARCH_SPAN * 2),
            SEARCH_SPAN
        ),
        slice(
            mu_init - SEARCH_SPAN,
            mu_init + (SEARCH_SPAN * 2),
            SEARCH_SPAN
        ),
        slice(
            max([sigma_init - (SEARCH_SPAN * 4), 4]),
            sigma_init + (SEARCH_SPAN * 2 * 4),
            (SEARCH_SPAN * 4)
        ),
        slice(NOISE_MIN, NOISE_MAX, NOISE_STEP),  # noise slice
        slice(BIAS_DRIFT_MIN, 1, BIAS_DRIFT_STEP),  # bias_drift_range_min
        slice(1, BIAS_DRIFT_MAX, BIAS_DRIFT_STEP),  # bias_drift_range_max
        slice(
            BIAS_DRIFT_OSCILLATION_MIN,
            BIAS_DRIFT_OSCILLATION_MAX,
            BIAS_DRIFT_OSCILLATION_STEP
        )  # bias_drift_oscillations
    )

    input_names = [
        "SPECIAL_CONTROLS_CRITERIA",
        "SEARCH_SPAN",
        "BIAS_CATEGORY",
        "BIAS_MIN",
        "BIAS_MAX",
        "BIAS_DRIFT_MIN",
        "BIAS_DRIFT_MAX",
        "BIAS_DRIFT_STEP",
        "BIAS_DRIFT_OSCILLATION_MIN",
        "BIAS_DRIFT_OSCILLATION_MAX",
        "BIAS_DRIFT_OSCILLATION_STEP",
        "NOISE_MIN",
        "NOISE_MAX",
        "NOISE_STEP",
    ]

    input_df = pd.DataFrame(
        [
            str(SPECIAL_CONTROLS_CRITERIA),
            SEARCH_SPAN,
            BIAS_CATEGORY,
            BIAS_MIN,
            BIAS_MAX,
            BIAS_DRIFT_MIN,
            BIAS_DRIFT_MAX,
            BIAS_DRIFT_STEP,
            BIAS_DRIFT_OSCILLATION_MIN,
            BIAS_DRIFT_OSCILLATION_MAX,
            BIAS_DRIFT_OSCILLATION_STEP,
            NOISE_MIN,
            NOISE_MAX,
            NOISE_STEP,
        ],
        columns=["icgmSensorResults"],
        index=input_names
    )

    return rranges, input_df


def str2bool(string_):
    return string_.lower() in ("yes", "true", "t", "1")


def input_table_to_dict(input_df):
    dict_ = dict()

    # first parse and format the settings
    all_settings = input_df["settings"].dropna()
    dict_["settings_dictionary"] = all_settings.to_dict()

    for k in dict_["settings_dictionary"].keys():
        if k in [
            "dynamic_carb_absorption_enabled",
            "retrospective_correction_enabled"
        ]:

            dict_["settings_dictionary"][k] = str2bool(
                dict_["settings_dictionary"][k]
            )
        else:
            dict_["settings_dictionary"][k] = np.safe_eval(
                dict_["settings_dictionary"][k]
            )
    if "suspend_threshold" not in dict_["settings_dictionary"].keys():
        dict_["settings_dictionary"]["suspend_threshold"] = None

    # then parse and format the rest
    input_df_T = (
        input_df.drop(columns=["settings"]).dropna(axis=0, how="all").T
    )

    input_df_columns = input_df_T.columns
    for col in input_df_columns:
        if "units" in col:
            dict_[col] = input_df_T[col].dropna().unique()[0]
        elif "offset" in col:
            dict_[col] = int(np.safe_eval(input_df_T[col].dropna()[0]))
        elif "time_to_calculate" in col:
            dict_[col] = (
                datetime.datetime.fromisoformat(
                    pd.to_datetime(input_df_T[col].dropna()[0]).isoformat()
                )
            )
        else:
            temp_df = input_df_T[col].dropna()
            temp_array = []
            for v in temp_df.values:
                if ":" in v:
                    if len(v) == 7:
                        obj = (
                            datetime.time.fromisoformat(
                                pd.to_datetime(v).strftime("%H:%M:%S")
                            )
                        )
                    elif len(v) == 8:
                        obj = datetime.time.fromisoformat(v)
                    elif len(v) > 8:
                        obj = (
                            datetime.datetime.fromisoformat(
                                pd.to_datetime(v).isoformat()
                            )
                        )
                    else:
                        obj = np.safe_eval(v)
                elif "DoseType" in v:
                    obj = DoseType.from_str(v[9:])
                else:
                    obj = np.safe_eval(v)

                temp_array = np.append(temp_array, obj)

            dict_[col] = list(temp_array)

    return dict_


# create pandas dataframes from the input data
def dict_inputs_to_dataframes(input_data):
    # define the dataframes to store the data in
    df_basal_rate = pd.DataFrame()
    df_carb = pd.DataFrame()
    df_carb_ratio = pd.DataFrame()
    df_dose = pd.DataFrame()
    df_glucose = pd.DataFrame()
    df_last_temporary_basal = pd.DataFrame()
    df_misc = pd.DataFrame()
    df_sensitivity_ratio = pd.DataFrame()
    df_settings = pd.DataFrame()
    df_target_range = pd.DataFrame()

    for k in input_data.keys():
        if type(input_data[k]) != dict:
            if "basal_rate" in k:
                df_basal_rate[k] = input_data.get(k)
            elif "carb_ratio" in k:
                df_carb_ratio[k] = input_data.get(k)
            elif "carb" in k:
                df_carb[k] = input_data.get(k)
            elif "dose" in k:
                df_dose[k] = input_data.get(k)
            elif "glucose" in k:
                df_glucose[k] = input_data.get(k)
            elif "last_temporary_basal" in k:
                # TODO: change how this is dealt with in pyloopkit
                df_last_temporary_basal[k] = input_data.get(k)
            elif "sensitivity_ratio" in k:
                df_sensitivity_ratio[k] = input_data.get(k)
            elif "target_range" in k:
                df_target_range[k] = input_data.get(k)
            else:
                if np.size(input_data.get(k)) == 1:
                    if type(input_data[k]) == list:
                        df_misc.loc[k, 0] = input_data.get(k)[0]
                    else:
                        df_misc.loc[k, 0] = input_data.get(k)
        else:
            if "settings_dictionary" in k:
                settings_dictionary = input_data.get("settings_dictionary")
                for sk in settings_dictionary.keys():
                    if np.size(settings_dictionary.get(sk)) == 1:
                        if type(settings_dictionary[sk]) == list:
                            df_settings.loc[sk, "settings"] = (
                                settings_dictionary.get(sk)[0]
                            )
                        else:
                            df_settings.loc[sk, "settings"] = (
                                settings_dictionary.get(sk)
                            )
                    else:
                        if sk in ["model", "default_absorption_times"]:
                            # TODO: change this in the loop algorithm
                            # to take 2 to 3 inputs instead of 1
                            df_settings.loc[sk, "settings"] = (
                                str(settings_dictionary.get(sk))
                            )

    return (
        df_basal_rate, df_carb, df_carb_ratio, df_dose, df_glucose,
        df_last_temporary_basal, df_misc, df_sensitivity_ratio,
        df_settings, df_target_range
    )