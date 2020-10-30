import pickle
import numpy as np
from pytest import approx
from scipy.stats import johnsonsu
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
    sensor_properties = sensor_generator.individual_sensor_properties
    for s in range(batch_size):
        sensor = sensor_generator.sensors[s]
        sensor_generator.individual_sensor_properties

        # check that the sensor characteristics are being passed properly
        assert (
            sensor.initial_bias == sensor_properties.loc[s, "initial_bias"].values[0]
            and sensor.noise_per_sensor == sensor_properties.loc[s, "noise_per_sensor"].values[0]
            and sensor.phi_drift == sensor_properties.loc[s, "phi_drift"].values[0]
            and sensor.bias_drift_range_start == sensor_properties.loc[s, "bias_drift_range_start"].values[0]
            and sensor.bias_drift_range_end == sensor_properties.loc[s, "bias_drift_range_end"].values[0]
            and sensor.bias_drift_oscillations == sensor_properties.loc[s, "bias_drift_oscillations"].values[0]
            and sensor.bias_norm_factor == sensor_properties.loc[s, "bias_norm_factor"].values[0]
            and sensor.noise_coefficient == sensor_properties.loc[s, "noise_coefficient"].values[0]
            and sensor.delay_minutes == sensor_properties.loc[s, "delay"].values[0]
            and sensor.random_seed == sensor_properties.loc[s, "random_seed"].values[0]
            and sensor.bias_drift_type == sensor_properties.loc[s, "bias_drift_type"].values[0]
        )

        # check that the resulting noise added fits within the tolerance of the noise per sensor
        assert sensor.noise_per_sensor == approx(np.std(sensor.noise), rel=1e-1)

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

        assert (sensor.initial_bias >= initial_bias_min) and (sensor.initial_bias <= initial_bias_max)

        # check that the bias drift is applied correctly
        # there are a few use cases:
        # * when oscillations >= 2 then the drift should cover the full range from bias_drift_range_start to bias_drift_range_end
        # * when osciallations < 2 then the drift will be highly dependent on phi and the range of any one sensor will be limited
        # NOTE: if you want to see how this works, see the plots commented below
        bias_drift_range_start = sensor_properties.loc[0, "bias_drift_range_start"].values[0]
        bias_drift_range_end = sensor_properties.loc[0, "bias_drift_range_end"].values[0]
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


def test_that_results_are_repeatable():
    """
    check that fit gives the same exact results as the fit to the benchmark dataset
    see (/notebooks/make_benchmark_datasets.py for details on the benchmark dataset)
    """

    benchmark_sensor_generator_obj = pickle.load(open("benchmark_results_2020_10_30.pkl", "rb"))

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
    new_sensor_generator.fit(benchmark_sensor_generator_obj.true_bg_trace)

    assert new_sensor_generator.sc_thresholds == benchmark_sensor_generator_obj.sc_thresholds
    assert new_sensor_generator.batch_training_size == benchmark_sensor_generator_obj.batch_training_size
    assert (
        new_sensor_generator.johnson_parameter_search_range
        == benchmark_sensor_generator_obj.johnson_parameter_search_range
    )
    assert new_sensor_generator.individual_sensor_properties.equals(
        benchmark_sensor_generator_obj.individual_sensor_properties
    )
    assert new_sensor_generator.icgm_special_controls_accuracy_table.equals(
        benchmark_sensor_generator_obj.icgm_special_controls_accuracy_table
    )
