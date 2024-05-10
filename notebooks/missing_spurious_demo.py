import numpy as np
import matplotlib.pyplot as plt
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator

n_sensors = 5
batch_training_size=1
x = np.arange(0, 2880) / 2880 * 5 * np.pi
y = x * np.sin(x)
a = (400-40) / (max(y) - min(y))

# True BG trace that spans 40-400 in an interesting way
true_bg_trace = y * a + 40 - min(y) * a

plt.plot(true_bg_trace)
plt.show()

#%% Should be the same as before:
sensor_generator_1 = iCGMSensorGenerator(batch_training_size=batch_training_size, spurious_missing=False)
sensor_generator_1.fit(true_bg_trace)
sensor_generator_1.generate_sensors(n_sensors, sensor_start_datetime=0)
plt.plot(sensor_generator_1.icgm_traces.T)
plt.show()

# Check that there are no missing values
np.isnan(sensor_generator_2.icgm_traces).sum()

#%% Now with missing and spurious values:
sensor_generator_2 = iCGMSensorGenerator(batch_training_size=batch_training_size, spurious_missing=True)
sensor_generator_2.fit(true_bg_trace)
sensor_generator_2.generate_sensors(n_sensors, sensor_start_datetime=0)
plt.plot(sensor_generator_2.icgm_traces.T)
plt.show()

# Check that there are missing values
np.isnan(sensor_generator_2.icgm_traces).sum()

#%% play with other parameters
sensor_generator_3 = iCGMSensorGenerator(batch_training_size=batch_training_size, spurious_missing=True,
                                         avg_normal_time=120, avg_missing_time=120, p_spurious_missing=1)
sensor_generator_3.fit(true_bg_trace)
sensor_generator_3.generate_sensors(n_sensors, sensor_start_datetime=0)
plt.plot(sensor_generator_2.icgm_traces.T)
plt.show()

# Check that there are more missing values
np.isnan(sensor_generator_3.icgm_traces).sum()