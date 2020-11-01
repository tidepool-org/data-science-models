import pickle
import os
import copy
import datetime
import numpy as np
from pytest import approx
from scipy.stats import johnsonsu
from tidepool_data_science_models.models.icgm_sensor import iCGMSensor
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
from tidepool_data_science_models.models.icgm_sensor_generator_functions import create_dataset


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
        open("benchmark_results_with_df_sensor_properties_2020_10_30.pkl", "rb")
    )

    new_benchmark_sensor_generator_obj = pickle.load(
        open("benchmark_results_with_new_dict_sensor_properties_2020_11_01.pkl", "rb")
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
    assert np.array_equal(new_benchmark_sensor_generator_obj.icgm_traces, benchmark_sensor_generator_obj.icgm_traces)


def test_that_results_are_repeatable_before_after_sensor_property_refactor():
    """
    check that fit gives the same exact results as the fit to the benchmark dataset
    see (/notebooks/make_benchmark_datasets.py for details on the benchmark dataset)
    """

    benchmark_sensor_generator_obj = pickle.load(
        open("benchmark_results_with_df_sensor_properties_2020_10_30.pkl", "rb")
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
    assert np.array_equal(new_sensor_generator.icgm_traces, benchmark_sensor_generator_obj.icgm_traces)

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
        open(os.path.join("..", "tests", "benchmark_results_with_new_dict_sensor_properties_2020_11_01.pkl"), "rb")
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
