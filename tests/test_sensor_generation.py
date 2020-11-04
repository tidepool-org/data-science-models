import pickle
import os
import copy
import datetime
import numpy as np
from pytest import approx
from scipy.stats import johnsonsu
from tidepool_data_science_models.models.icgm_sensor import iCGMSensor
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
from tidepool_data_science_models.models.icgm_sensor_generator_functions import (
    create_dataset,
    generate_icgm_sensors,
    calc_icgm_sc_table,
    preprocess_data,
)


def test_that_fit_icgm_sensor_has_correct_stats():
    """
    Fit an icgm sensor and then check that resulting sensors have expected characteristics
    """
    df, _ = create_dataset(
        kind="sine",
        N=288 * 10,
        min_value=40,
        max_value=400,
        time_interval=5,
        flat_value=np.nan,
        oscillations=1,
        random_seed=0,
    )

    batch_size = 3
    sensor_generator = iCGMSensorGenerator(
        sc_thresholds=None,  # This is required only for iCGM sensors for now (A-G)
        batch_training_size=batch_size,
        use_g6_accuracy_in_loss=False,
        bias_type="percentage_of_value",
        bias_drift_type="random",
        random_seed=0,
        verbose=False,
        brute_force_search_range=(
            slice(0, 1, 1),  # a of johnsonsu (fixed to 0)
            slice(1, 2, 1),  # b of johnsonsu (fixed to 1)
            slice(-7, 8, 1),  # mu of johnsonsu
            slice(1, 8, 1),  # sigma of johnsonsu
            slice(0, 11, 1),  # max allowable sensor noise in batch of sensors
            slice(0.9, 1, 1),  # setting bias drift min (fixed to 0.9)
            slice(1.1, 1.3, 1),  # setting bias drift min (fixed to 1.1)
            slice(1, 2, 1),  # setting bias drift oscillations (fixed to 1)
        ),
    )

    sensor_generator.fit(df["value"].values)

    # check that the sensor values used in the fit are written to the sensor output
    sensor_properties = sensor_generator.sensor_properties
    for s in range(batch_size):
        sensor = sensor_generator.sensors[s]

        # check that the sensor characteristics are being passed properly
        for key in sensor.sensor_properties.keys():
            assert np.array_equal(sensor.sensor_properties[key], sensor_properties[key][s])

        # check that the resulting noise added fits within the tolerance of the noise per sensor
        assert sensor.sensor_properties["noise_per_sensor"] == approx(
            np.std(sensor.sensor_properties["noise"]), rel=1e-1
        )

        # check that the initial bias fits within the fit distribution
        initial_bias_min = johnsonsu.ppf(
            0.001,
            a=sensor_generator.dist_params[0],
            b=sensor_generator.dist_params[1],
            loc=sensor_generator.dist_params[2],
            scale=sensor_generator.dist_params[3],
        )

        initial_bias_max = johnsonsu.ppf(
            0.999,
            a=sensor_generator.dist_params[0],
            b=sensor_generator.dist_params[1],
            loc=sensor_generator.dist_params[2],
            scale=sensor_generator.dist_params[3],
        )

        assert (sensor.sensor_properties["initial_bias"] >= initial_bias_min) and (
            sensor.sensor_properties["initial_bias"] <= initial_bias_max
        )

        # check that the bias drift is applied correctly
        # there are a few use cases:
        # * when oscillations >= 2 then the drift should cover the full range from bias_drift_range_start to bias_drift_range_end
        # * when osciallations < 2 then the drift will be highly dependent on phi and the range of any one sensor will be limited
        # NOTE: if you want to see how this works, see the plots commented below
        bias_drift_range_start = sensor_properties["bias_drift_range_start"][0]
        bias_drift_range_end = sensor_properties["bias_drift_range_end"][0]

        for phi in np.arange(-np.pi, np.pi, np.pi / 2):
            for bias_drift_oscillations in [1 / 32, 1, 2]:
                t = np.linspace(0, (bias_drift_oscillations * np.pi), 2880)
                sn = np.sin(t + phi)
                drift_multiplier = np.interp(sn, (-1, 1), (bias_drift_range_start, bias_drift_range_end,),)

                drift_multiplier_min = drift_multiplier.min()
                drift_multiplier_max = drift_multiplier.max()
                if bias_drift_oscillations < 2:
                    assert (drift_multiplier_min >= bias_drift_range_start) and (
                        drift_multiplier_max <= bias_drift_range_end
                    )
                else:
                    assert bias_drift_range_start == approx(drift_multiplier_min, abs=1e-2)
                    assert bias_drift_range_end == approx(drift_multiplier_max, abs=1e-2)

                # import matplotlib.pyplot as plt
                # plt.plot(drift_multiplier)
                # plt.title("Bias Drift Applied with phi={:.2f}, oscillations={}".format(phi, bias_drift_oscillations))
                # plt.show()


def test_that_results_are_same_before_after_sensor_property_refactor():
    """
    check that fit gives the same exact results as the fit to the benchmark dataset
    see (/notebooks/make_benchmark_datasets.py for details on the benchmark dataset)
    """

    benchmark_sensor_generator_obj = pickle.load(
        open(os.path.join(".", "tests", "benchmark_results_with_df_sensor_properties_2020_10_30.pkl"), "rb")
    )

    new_benchmark_sensor_generator_obj = pickle.load(
        open(os.path.join(".", "tests", "benchmark_results_with_new_dict_sensor_properties_2020_11_01.pkl"), "rb")
    )

    assert new_benchmark_sensor_generator_obj.sc_thresholds == benchmark_sensor_generator_obj.sc_thresholds
    assert new_benchmark_sensor_generator_obj.batch_training_size == benchmark_sensor_generator_obj.batch_training_size
    assert (
        new_benchmark_sensor_generator_obj.johnson_parameter_search_range
        == benchmark_sensor_generator_obj.johnson_parameter_search_range
    )

    assert new_benchmark_sensor_generator_obj.icgm_special_controls_accuracy_table.equals(
        benchmark_sensor_generator_obj.icgm_special_controls_accuracy_table
    )

    # test that the same icgm traces are generated
    assert np.array_equal(
        np.round(new_benchmark_sensor_generator_obj.icgm_traces), np.round(benchmark_sensor_generator_obj.icgm_traces)
    )


def test_that_results_are_repeatable_before_after_sensor_property_refactor():
    """
    check that fit gives the same exact results as the fit to the benchmark dataset
    see (/notebooks/make_benchmark_datasets.py for details on the benchmark dataset)
    """

    benchmark_sensor_generator_obj = pickle.load(
        open(os.path.join(".", "tests", "benchmark_results_with_df_sensor_properties_2020_10_30.pkl"), "rb")
    )

    batch_size = 3
    random_seed = 0
    new_sensor_generator = iCGMSensorGenerator(
        batch_training_size=batch_size,
        use_g6_accuracy_in_loss=False,
        bias_type="percentage_of_value",
        bias_drift_type="random",
        random_seed=random_seed,
        verbose=False,
        brute_force_search_range=(
            slice(0, 1, 1),  # a of johnsonsu (fixed to 0)
            slice(1, 2, 1),  # b of johnsonsu (fixed to 1)
            slice(-7, 8, 1),  # mu of johnsonsu
            slice(1, 8, 1),  # sigma of johnsonsu
            slice(0, 11, 1),  # max allowable sensor noise in batch of sensors
            slice(0.9, 1, 1),  # setting bias drift min (fixed to 0.9)
            slice(1.1, 1.3, 1),  # setting bias drift min (fixed to 1.1)
            slice(1, 2, 1),  # setting bias drift oscillations (fixed to 1)
        ),
    )

    # refit to the same benchmark true bg dataset
    benchmark_true_bg_trace = benchmark_sensor_generator_obj.true_bg_trace
    new_sensor_generator.fit(benchmark_true_bg_trace)

    assert new_sensor_generator.sc_thresholds == benchmark_sensor_generator_obj.sc_thresholds
    assert new_sensor_generator.batch_training_size == benchmark_sensor_generator_obj.batch_training_size
    assert (
        new_sensor_generator.johnson_parameter_search_range
        == benchmark_sensor_generator_obj.johnson_parameter_search_range
    )

    assert new_sensor_generator.icgm_special_controls_accuracy_table.equals(
        benchmark_sensor_generator_obj.icgm_special_controls_accuracy_table
    )

    # test that the same icgm traces are generated
    assert np.array_equal(
        np.round(new_sensor_generator.icgm_traces), np.round(benchmark_sensor_generator_obj.icgm_traces)
    )

    # this assertion is now changed given that individual_sensor_properties have been changed from df to dict,
    # BUT, the icgm traces would only be exactly equal if all of the sensor properties were identical,
    # which is the test above.

    # assert new_sensor_generator.individual_sensor_properties.equals(
    #     benchmark_sensor_generator_obj.individual_sensor_properties
    # )


def test_given_same_true_bg_trace_and_sensor_properties_create_new_sensor_and_get_same_icgm_trace():
    """
    given the same true bg trace and sensor characteristics, we should be able to recreate the same icgm trace with
    the update function
    """

    benchmark_sensor_generator_obj = pickle.load(
        open(os.path.join(".", "tests", "benchmark_results_with_new_dict_sensor_properties_2020_11_01.pkl"), "rb")
    )

    benchmark_sensor = benchmark_sensor_generator_obj.sensors[0]
    benchmark_true_bg_trace = benchmark_sensor_generator_obj.true_bg_trace

    new_sensor_properties = copy.deepcopy(benchmark_sensor.sensor_properties)

    # delay the noise and drift multiplier by the delay amount to get identical traces
    n_sensor_delay_datapoints = int(benchmark_sensor.sensor_properties["delay"] / 5)
    new_sensor_properties["noise"] = np.append(
        [np.nan] * n_sensor_delay_datapoints, benchmark_sensor.sensor_properties["noise"]
    )
    new_sensor_properties["drift_multiplier"] = np.append(
        [np.nan] * n_sensor_delay_datapoints, benchmark_sensor.sensor_properties["drift_multiplier"]
    )

    # create a new sensor with the benchmark sensor properties
    sensor_datetime = datetime.datetime(2020, 1, 1)
    new_sensor = iCGMSensor(current_datetime=sensor_datetime, sensor_properties=new_sensor_properties)

    # this is the case where we are passing in all of the sensor characteristics
    for expected_time_index, true_bg_val in enumerate(benchmark_true_bg_trace):
        new_sensor.update(sensor_datetime, patient_true_bg=true_bg_val)
        if expected_time_index > 1:
            # check that the icgm values are identical
            assert new_sensor.current_sensor_bg == benchmark_sensor_generator_obj.icgm_traces[0, expected_time_index]
            # NOTE: I am leaving this in here if anyone wants to view the output
            # print(
            #     expected_time_index,
            #     sensor_datetime,
            #     true_bg_val,
            #     new_sensor.current_sensor_bg,
            #     benchmark_sensor_generator_obj.icgm_traces[0, expected_time_index],
            #     new_sensor.sensor_properties["noise"][expected_time_index],
            #     benchmark_sensor_generator_obj.sensor_properties["noise"][0, expected_time_index],
            #     new_sensor.sensor_properties["drift_multiplier"][expected_time_index],
            #     benchmark_sensor_generator_obj.sensor_properties["drift_multiplier"][0, expected_time_index],
            #     new_sensor.sensor_properties["bias_factor"],
            #     benchmark_sensor_generator_obj.sensor_properties["bias_factor"][0],
            # )
            sensor_datetime += datetime.timedelta(minutes=5)


def test_recreate_icgm_sensor_and_values_with_same_random_seed():
    # we should be able to create a sensor with iCGMSensor and then be able to recreate using the same random seed
    # create a new sensor with the benchmark sensor properties

    df, _ = create_dataset(
        kind="sine",
        N=288 * 10,
        min_value=40,
        max_value=400,
        time_interval=5,
        flat_value=np.nan,
        oscillations=1,
        random_seed=0,
    )

    true_bg_trace = df["value"].values

    sensor_datetime = datetime.datetime(2020, 1, 1)

    # NOTE: random seed is not specified, but rather is gathered from the sensor instantiation
    first_sensor = iCGMSensor(
        current_datetime=sensor_datetime,
        sensor_properties={
            "bias_type": "percentage_of_value",
            "initial_bias": 3,
            "bias_norm_factor": 55,
            "bias_drift_type": "random",
            "phi_drift": 0,
            "bias_drift_range_start": 0.9,
            "bias_drift_range_end": 1.1,
            "bias_drift_oscillations": 1,
            "noise_per_sensor": 7,
            "delay": 10,
        },
    )

    # make an icgm trace using the udpate function
    for expected_time_index, true_bg_val in enumerate(true_bg_trace):
        first_sensor.update(sensor_datetime, patient_true_bg=true_bg_val)
        sensor_datetime += datetime.timedelta(minutes=5)

    sensor_datetime = datetime.datetime(2020, 1, 1)
    second_sensor = iCGMSensor(
        current_datetime=sensor_datetime, sensor_properties=copy.deepcopy(first_sensor.sensor_properties)
    )

    for expected_time_index, true_bg_val in enumerate(true_bg_trace):
        second_sensor.update(sensor_datetime, patient_true_bg=true_bg_val)
        sensor_datetime += datetime.timedelta(minutes=5)

    assert np.array_equal(first_sensor.sensor_bg_history[2:], second_sensor.sensor_bg_history[2:])


def test_icgm_sensor_generation_mimics_icgm_sensitivity_analysis_behavior():
    """ Mimic the way the iCGMSensor is going to work in the icgm sensitivity analyis
    [x] make sure that each sensor in a batch of sensors for a virtual patient is different
    [x] make sure that temp_basal_only sensors are identical to correction_bolus and meal_bolus sensors
    """

    number_of_virtual_patients_to_test = 10
    number_of_sensors_in_a_batch = 30
    sensor_datetime = datetime.datetime(2020, 1, 1)
    all_virtual_patients = dict()

    test_sensor_properties = {
        "bias_type": "percentage_of_value",
        "a": 0,
        "b": 1,
        "mu": 2,
        "sigma": 1,
        "bias_drift_type": "random",
        "bias_drift_range_start": 0.9,
        "bias_drift_range_end": 1.1,
        "bias_drift_oscillations": 1,
        "noise_coefficient": 5,
        "delay": 10,
    }

    # first create the sensors in a manner similar to the icgm sensitivity analysis
    for vp_index in range(number_of_virtual_patients_to_test):
        for bg_condition_index in range(1, 10):
            for sensor_in_batch_index in range(number_of_sensors_in_a_batch):
                experiment_type = "temp_basal_only"
                virtual_patient_dict = dict()
                temp_basal_sim_id = "vp{}.bg{}.s{}.{}".format(
                    vp_index, bg_condition_index, sensor_in_batch_index, experiment_type
                )
                virtual_patient_dict["sim_id"] = temp_basal_sim_id
                virtual_patient_dict["start_time"] = sensor_datetime
                virtual_patient_dict["patient"] = {"name": "Virtual Patient"}
                icgm_sensor = iCGMSensor(current_datetime=sensor_datetime, sensor_properties=test_sensor_properties,)
                virtual_patient_dict["patient"]["sensor"] = copy.deepcopy(icgm_sensor.sensor_properties)
                all_virtual_patients[temp_basal_sim_id] = virtual_patient_dict

                for experiment_type in ["correction_bolus", "meal_bolus"]:
                    virtual_patient_dict = dict()
                    sim_id_name = "vp{}.bg{}.s{}.{}".format(
                        vp_index, bg_condition_index, sensor_in_batch_index, experiment_type
                    )
                    virtual_patient_dict["sim_id"] = sim_id_name
                    virtual_patient_dict["start_time"] = sensor_datetime
                    virtual_patient_dict["patient"] = {"name": "Virtual Patient"}
                    icgm_sensor = iCGMSensor(
                        current_datetime=sensor_datetime,
                        sensor_properties=all_virtual_patients[temp_basal_sim_id]["patient"]["sensor"],
                    )
                    virtual_patient_dict["patient"]["sensor"] = copy.deepcopy(icgm_sensor.sensor_properties)
                    all_virtual_patients[sim_id_name] = virtual_patient_dict

    # next, make sure that the sensor properties are different for each sensor in the batch,
    # but the same between experiment types
    for vp_index in range(number_of_virtual_patients_to_test):
        for bg_condition_index in range(1, 10):
            for sensor_in_batch_index in range(number_of_sensors_in_a_batch):
                sim_id_name = "vp{}.bg{}.s{}.{}".format(
                    vp_index, bg_condition_index, sensor_in_batch_index, "temp_basal_only"
                )
                sensor_b_properties = all_virtual_patients[sim_id_name]["patient"]["sensor"]

                # compare sensors in a batch to make sure that are different
                for compare_sensor_in_batch_index in range(number_of_sensors_in_a_batch):
                    if sensor_in_batch_index != compare_sensor_in_batch_index:
                        sim_id_name_b_compare = "vp{}.bg{}.s{}.{}".format(
                            vp_index, bg_condition_index, compare_sensor_in_batch_index, "temp_basal_only"
                        )
                        compare_sensor_b_properties = all_virtual_patients[sim_id_name_b_compare]["patient"]["sensor"]
                        for key in sensor_b_properties.keys():
                            assert ~np.array_equal(sensor_b_properties[key], compare_sensor_b_properties[key])

                # compare correction_bolus and meal_bolus with temp_basal_only to make sure they are the same
                for experiment_type in ["correction_bolus", "meal_bolus"]:
                    sim_id_name_b_compare = "vp{}.bg{}.s{}.{}".format(
                        vp_index, bg_condition_index, sensor_in_batch_index, experiment_type
                    )
                    compare_sensor_b_properties = all_virtual_patients[sim_id_name_b_compare]["patient"]["sensor"]
                    for key in sensor_b_properties.keys():
                        assert np.array_equal(sensor_b_properties[key], compare_sensor_b_properties[key])


def test_number_of_spurious_events_is_correct():

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

    # import matplotlib.pyplot as plt
    # plt.plot(true_bg_trace)
    # plt.plot(icgm_sensor.sensor_bg_history)
    # plt.show()


def test_that_spurious_event_values_are_always_within_icgm_special_controls():
    # %% test that the value of the spurious events is within icgm special controls
    true_df, _ = create_dataset(kind="linear", N=288 * 10, min_value=40, max_value=400, time_interval=5, random_seed=0,)
    true_bg_trace = true_df["value"].values

    icgm_trace, ind_sensor_properties = generate_icgm_sensors(
        true_bg_trace,
        [0, 1, 0, 1],  # [a, b, mu, sigma]
        n_sensors=1,  # (suggest 100 for speed, 1000 for thoroughness)
        bias_type="percentage_of_value",  # (constant_offset, percentage_of_value)
        bias_drift_type="none",  # options (none, linear, random)
        delay=10,  # (suggest 0, 5, 10, 15)
        max_number_of_spurious_events_per_sensor_life=1000,
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


def test_that_batch_sensors_fit_icgm_special_controls_with_spurious_events():
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

    batch_training_size = 3
    sensor_generator = iCGMSensorGenerator(
        batch_training_size=batch_training_size, max_number_of_spurious_events_per_sensor_life=100,
    )
    sensor_generator.fit(true_bg_trace)

    # make sure that the number of spurious events is less than or equal to the max allowable
    assert sensor_generator.sensor_properties["spurious"].sum() <= np.sum(
        sensor_generator.sensor_properties["max_number_of_spurious_events_per_sensor_life"]
    )

    # make sure that the value of the spurious events is within icgm special controls
    assert sensor_generator.icgm_special_controls_accuracy_table.loc["H", "icgmSensorResults"] == 100
    assert sensor_generator.icgm_special_controls_accuracy_table.loc["I", "icgmSensorResults"] == 100

    # make sure that each sensor has the right number of spurious events
    for s in sensor_generator.sensors:
        assert s.sensor_properties["spurious_events_per_sensor"] == s.sensor_properties["spurious"].sum()
        assert (
            s.sensor_properties["spurious"].sum()
            <= s.sensor_properties["max_number_of_spurious_events_per_sensor_life"]
        )
