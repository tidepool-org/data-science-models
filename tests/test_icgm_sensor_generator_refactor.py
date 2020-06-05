"""
Tests to confirm Jason's iCGM Generator Refactor returns the same output as the old code
"""

# %% Imports
import pandas as pd
import numpy as np
import datetime
import copy
import pytest
from tidepool_data_science_models.models.icgm_sensor_generator_OLD import icgm_simulator_old
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator, iCGMSensor
import tidepool_data_science_models.models.icgm_sensor_generator_functions as sf

# %% Tests


def test_original_function_default_state():
    """
    Test original functionality works between multiple calls (static random state with no input)
    """
    icgm_traces_1, individual_sensor_properties_1, batch_sensor_properties_1 = icgm_simulator_old()
    icgm_traces_2, individual_sensor_properties_2, batch_sensor_properties_2 = icgm_simulator_old()

    assert np.array_equal(icgm_traces_1, icgm_traces_2)
    assert individual_sensor_properties_1.equals(individual_sensor_properties_2)
    assert batch_sensor_properties_1.equals(batch_sensor_properties_2)


def test_refactor_default_state():
    """
    Test refactor functionality works returns the same as original output)
    """

    # Get original results
    original_icgm_traces, original_individual_sensor_properties, original_batch_sensor_properties = icgm_simulator_old()

    # Calculated refactored results
    test_bg_trace = sf.generate_test_bg_trace(days_of_data=2)

    icgm_sensor_generator = iCGMSensorGenerator(sensor_batch_size=3, true_dataset_name="48hours-sinusoid")
    icgm_sensor_generator.fit(test_bg_trace)
    sensors = icgm_sensor_generator.generate_sensors(3)

    refactored_icgm_traces = icgm_sensor_generator.icgm_traces
    refactored_individual_sensor_properties, refactored_batch_sensor_properties = sf.calculate_sensor_generator_tables(
        icgm_sensor_generator
    )

    assert np.array_equal(refactored_icgm_traces, original_icgm_traces)
    assert refactored_individual_sensor_properties.equals(original_individual_sensor_properties)
    assert refactored_batch_sensor_properties.equals(original_batch_sensor_properties)


def create_sample_sensor(
    sensor_life_days, sensor_time,
):

    sample_sensor_properties = pd.DataFrame(index=[0])
    sample_sensor_properties["initial_bias"] = 1.992889
    sample_sensor_properties["phi_drift"] = 2.158842
    sample_sensor_properties["bias_drift_range_start"] = 0.835931
    sample_sensor_properties["bias_drift_range_end"] = 1.040707
    sample_sensor_properties["bias_drift_oscillations"] = 1.041129
    sample_sensor_properties["bias_norm_factor"] = 55.000000
    sample_sensor_properties["noise_coefficient"] = 7.195753
    sample_sensor_properties["delay"] = 10
    sample_sensor_properties["random_seed"] = 0
    sample_sensor_properties["bias_drift_type"] = "random"

    sample_sensor = iCGMSensor(
        sensor_properties=sample_sensor_properties,
        sensor_life_days=sensor_life_days,
        sensor_time=sensor_time,
    )

    return sample_sensor, sample_sensor_properties


def check_sensor_properties(sensor, sensor_properties):
    assert sensor.initial_bias == sensor_properties["initial_bias"].values[0]
    assert sensor.phi_drift == sensor_properties["phi_drift"].values[0]
    assert sensor.bias_drift_range_start == sensor_properties["bias_drift_range_start"].values[0]
    assert sensor.bias_drift_range_end == sensor_properties["bias_drift_range_end"].values[0]
    assert sensor.bias_drift_oscillations == sensor_properties["bias_drift_oscillations"].values[0]
    assert sensor.bias_norm_factor == sensor_properties["bias_norm_factor"].values[0]
    assert sensor.noise_coefficient == sensor_properties["noise_coefficient"].values[0]
    assert sensor.delay == sensor_properties["delay"].values[0]
    assert sensor.random_seed == sensor_properties["random_seed"].values[0]
    assert sensor.bias_drift_type == sensor_properties["bias_drift_type"].values[0]


def test_sample_sensor_creation():

    sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=10, sensor_time=0,)
    assert isinstance(sample_sensor, iCGMSensor)
    check_sensor_properties(sample_sensor, sample_sensor_properties)
    assert sample_sensor.sensor_time == 0
    assert sample_sensor.true_bg_history == []
    assert sample_sensor.sensor_bg_history == []
    assert sample_sensor.sensor_life_days == 10

    sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=1, sensor_time=287,)
    assert sample_sensor.sensor_time == 287
    assert sample_sensor.sensor_life_days == 1


def test_invalid_sensor_creation():

    with pytest.raises(Exception) as e:
        sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=1, sensor_time=288,)
    expected_exception_message = "Sensor time_index 288 outside of sensor life! "
    received_exception_message = str(e.value)
    assert expected_exception_message == received_exception_message


def test_wrong_sensor_backfill():
    """Sensor backfill should fail when backfilled data goes past the sensor start date"""
    sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=10, sensor_time=0)

    # original_sensor_state = copy.deepcopy(sample_sensor)
    backfill_data = [100, 101, 102, 103, 104]

    with pytest.raises(Exception) as e:
        sample_sensor.backfill_sensor_data(backfill_data)

    expected_exception_message = (
        "Sensor time_index -5 outside of sensor life! Trying to backfill data before start of sensor life. "
        + "Either establish the sensor at a different sensor_time or backfill with less data."
    )
    received_exception_message = str(e.value)
    assert expected_exception_message == received_exception_message


def test_correct_sensor_backfill():
    sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=10, sensor_time=5)

    # original_sensor_state = copy.deepcopy(sample_sensor)
    backfill_data = [100, 101, 102, 103, 104]
    sample_sensor.backfill_sensor_data(backfill_data)
    assert sample_sensor.sensor_time == 5

    expected_sensor_bg_history = [np.nan, np.nan, 85.53291217733658, 86.7455243131965, 91.30904749039524]
    np.testing.assert_equal(sample_sensor.sensor_bg_history, expected_sensor_bg_history)


def test_sensor_update():
    sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=10, sensor_time=0)
    for expected_sensor_time in range((10 * 288)-1):
        assert sample_sensor.sensor_time == expected_sensor_time
        _ = sample_sensor.get_bg(100)
        sample_sensor.update(None)
