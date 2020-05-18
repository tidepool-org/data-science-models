"""
Tests to confirm Jason's iCGM Generator Refactor returns the same output as the old code
"""

# %% Imports
import pandas as pd
import numpy as np
from src.models.icgm_sensor_generator_OLD import icgm_simulator_old
from src.models.icgm_sensor_generator import iCGMSensorGenerator, iCGMSensor
import src.models.icgm_sensor_generator_functions as sf

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


def test_create_icgm_sensor():

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

    sample_sensor = iCGMSensor(sensor_properties=sample_sensor_properties)

    # Add asserts
