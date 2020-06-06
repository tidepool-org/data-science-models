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


def create_sample_sensor(sensor_life_days, time_index, sensor_datetime=None):

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
        time_index=time_index,
        sensor_datetime=sensor_datetime,
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

    sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=10, time_index=0,)
    assert isinstance(sample_sensor, iCGMSensor)
    check_sensor_properties(sample_sensor, sample_sensor_properties)
    assert sample_sensor.time_index == 0
    assert sample_sensor.true_bg_history == []
    assert sample_sensor.sensor_bg_history == []
    assert sample_sensor.sensor_life_days == 10

    sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=1, time_index=287,)
    assert sample_sensor.time_index == 287
    assert sample_sensor.sensor_life_days == 1


def test_invalid_sensor_creation():
    """Test to make sure that invalid starting time_indexs are thrownthe correct Exceptions"""
    with pytest.raises(Exception) as e:
        sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=1, time_index=288,)

    # Reminder: Although there are 288 points / day, time_index starts at 0
    expected_exception_message = "Sensor time_index 288 outside of sensor life! "
    received_exception_message = str(e.value)
    assert expected_exception_message == received_exception_message

    with pytest.raises(Exception) as e:
        sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=1, time_index=-1,)
    expected_exception_message = "Sensor time_index -1 outside of sensor life! "
    received_exception_message = str(e.value)
    assert expected_exception_message == received_exception_message


def test_invalid_sensor_backfill():
    """Sensor backfill should fail when backfilled data goes past the sensor start date"""
    sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=10, time_index=0)

    # original_sensor_state = copy.deepcopy(sample_sensor)
    backfill_true_bg_history = [100, 101, 102, 103, 104]

    with pytest.raises(Exception) as e:
        sample_sensor.backfill_and_calculate_sensor_data(backfill_true_bg_history)

    expected_exception_message = (
        "Sensor time_index -5 outside of sensor life! Trying to backfill data before start of sensor life. "
        + "Either establish the sensor at a different time_index or backfill with less data."
    )
    received_exception_message = str(e.value)
    assert expected_exception_message == received_exception_message


def test_correct_sensor_backfill():
    sample_sensor, sample_sensor_properties = create_sample_sensor(sensor_life_days=10, time_index=5)

    # original_sensor_state = copy.deepcopy(sample_sensor)
    backfill_true_bg_history = [100, 101, 102, 103, 104]
    sample_sensor.backfill_and_calculate_sensor_data(backfill_true_bg_history)
    assert sample_sensor.time_index == 5

    expected_sensor_bg_history = [np.nan, np.nan, 93.6647980483592, 103.61317563648004, 101.79296809857229]
    assert str(sample_sensor.sensor_bg_history) == str(expected_sensor_bg_history)


def test_sensor_update():
    """Test sensor update() method functionality"""

    sensor_life_days = 10
    sensor_datetime = datetime.datetime(2020, 1, 1)
    expected_datetime_history = [sensor_datetime]

    sample_sensor, sample_sensor_properties = create_sample_sensor(
        sensor_life_days=sensor_life_days, time_index=0, sensor_datetime=sensor_datetime
    )

    _ = sample_sensor.get_bg(100)  #

    for expected_time_index in range(0, (10 * 288) - 1):
        assert sample_sensor.time_index == expected_time_index
        assert sample_sensor.sensor_datetime == sensor_datetime
        sensor_datetime += datetime.timedelta(minutes=5)
        expected_datetime_history.append(sensor_datetime)
        sample_sensor.update(sensor_datetime)
        _ = sample_sensor.get_bg(100)

    assert sample_sensor.time_index == 2879
    assert sample_sensor.sensor_datetime == datetime.datetime(2020, 1, 10, 23, 55)
    assert len(sample_sensor.datetime_history) == 10 * 288
    assert sample_sensor.datetime_history == expected_datetime_history

    with pytest.raises(Exception) as e:
        sample_sensor.update(None)

    expected_exception_message = "Sensor time_index 2880 outside of sensor life! Sensor has expired!"
    received_exception_message = str(e.value)
    assert expected_exception_message == received_exception_message


def test_get_loop_format():
    """Test that the sensor returns the proper loop format"""
    sensor_life_days = 10
    sensor_datetime = datetime.datetime(2020, 1, 1)
    sample_sensor, sample_sensor_properties = create_sample_sensor(
        sensor_life_days=sensor_life_days, time_index=5, sensor_datetime=sensor_datetime
    )

    backfill_true_bg_history = [100, 101, 102, 103, 104]
    sample_sensor.backfill_and_calculate_sensor_data(backfill_true_bg_history)

    expected_glucose_dates = [
        datetime.datetime(2019, 12, 31, 23, 35),
        datetime.datetime(2019, 12, 31, 23, 40),
        datetime.datetime(2019, 12, 31, 23, 45),
        datetime.datetime(2019, 12, 31, 23, 50),
        datetime.datetime(2019, 12, 31, 23, 55),
    ]
    expected_glucose_values = [400, 400, 94.0, 104.0, 102.0]  # NaNs are currently returned as 400s
    glucose_dates, glucose_values = sample_sensor.get_loop_format()

    assert glucose_dates == expected_glucose_dates
    assert glucose_values == expected_glucose_values


def test_backfill_calculations():
    """Compare normal sensor data updating to backfill calculation method"""

    true_bg_trace = [100, 101, 102, 103, 104]

    # Normal trace calculation starting at the start of sensor
    normal_sensor, _ = create_sample_sensor(sensor_life_days=10, time_index=0)
    for true_bg_value in true_bg_trace:
        normal_sensor.get_bg(true_bg_value)
        normal_sensor.update(None)

    backfilled_sensor, _ = create_sample_sensor(sensor_life_days=10, time_index=5)
    backfilled_sensor.backfill_and_calculate_sensor_data(true_bg_trace)

    assert str(normal_sensor.__dict__) == str(backfilled_sensor.__dict__)
