import numpy as np
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator

n_sensors = 3
true_bg_trace = np.tile(np.concatenate([np.arange(60, 201), np.flip(np.arange(60, 201))]), 10)
sensor_generator = iCGMSensorGenerator(batch_training_size=1)
sensor_generator.fit(true_bg_trace)
sensors = sensor_generator.generate_sensors(n_sensors, sensor_start_datetime=0)
print(sensor_generator.icgm_traces[0])

np.isnan(sensor_generator.icgm_traces).sum()
