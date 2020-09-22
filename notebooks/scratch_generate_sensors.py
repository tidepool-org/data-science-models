#%%
# import copy
from tidepool_data_science_simulator.models.patient_for_icgm_sensitivity_analysis import VirtualPatientISA
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
import time
import os
import datetime
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
import numpy as np
#%%
n_sensors = 10
true_bg_trace = np.arange(40, 140)
# true_bg_trace = sim_parser.patient_glucose_history.bg_values
sensor_generator.fit(true_bg_trace)
sensor_generator = iCGMSensorGenerator(batch_training_size=1)

sensor_generator.icgm_traces[0]