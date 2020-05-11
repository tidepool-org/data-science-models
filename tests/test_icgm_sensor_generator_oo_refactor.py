"""
Tests to confirm Jason's iCGM Generator Refactor returns the same output as the old code
"""

# %% Imports
import numpy as np
from src.models.icgm_sensor_generator_OLD import icgm_simulator
from src.models.icgm_sensor_generator import iCGMSensorGenerator

# %% Tests


def test_original_function_default_state():
    """
    Test original functionality works between multiple calls (static random state with no input)
    """
    icgm_traces_1, individual_sensor_properties_1, batch_sensor_properties_1 = icgm_simulator()
    icgm_traces_2, individual_sensor_properties_2, batch_sensor_properties_2 = icgm_simulator()

    assert np.array_equal(icgm_traces_1, icgm_traces_2)
    assert individual_sensor_properties_1.equals(individual_sensor_properties_2)
    assert batch_sensor_properties_1.equals(batch_sensor_properties_2)


def test_refactor_default_state():
    """
    Test refactor functionality works returns the same as original output)
    """

    icgm_sensor_generator = iCGMSensorGenerator(sensor_batch_size=3)
    icgm_sensor_generator.fit()
    icgm_sensor_generator.generate_sensors(3)
    icgm_sensor_generator.calculate_result_tables()

    original_icgm_traces, original_individual_sensor_properties, original_batch_sensor_properties = icgm_simulator()
    (
        new_icgm_traces,
        new_individual_sensor_properties,
        new_batch_sensor_properties,
    ) = icgm_sensor_generator.get_batch_results()

    assert np.array_equal(new_icgm_traces, original_icgm_traces)
    assert new_individual_sensor_properties.equals(original_individual_sensor_properties)
    assert new_batch_sensor_properties.equals(original_batch_sensor_properties)
