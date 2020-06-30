import numpy as np
import sys
import datetime
import copy


class SensorExpiredError(Exception):

    pass


# %% Definitions
class Sensor(object):
    """Base CGM Sensor Class"""

    def __init__(self):

        return

    def get_bg(self, true_bg):

        raise NotImplementedError

    def get_bg_trace(self, true_bg_trace):

        raise NotImplementedError

    def get_state(self):
        pass

    def update(self, time):
        # No state
        pass


class SensorState(object):

    def __init__(self, **kwargs):

        self.sensor_bg = kwargs.get("sensor_bg")
        self.sensor_bg_prediction = kwargs.get("sensor_bg_prediction")


class iCGMSensor(Sensor):
    """iCGM Sensor Object

    Parameters
        ----------
        sensor_properties : pandas DataFrame object
            A set of sensor properties needed to initialize an iCGM Sensor
        sensor_life_days : int
            The number of days the sensor will last.
        time_index : int
            The internal time of the sensor, used for errors and drift over the sensor life.
            (1 = 5 minutes, 2 = 10 minutes, etc)
        current_datetime : datetime.datetime or None
            The datetime timestamp associated with the time_index

    """

    def __init__(self, current_datetime, sensor_properties, sensor_life_days=10, time_index=0):

        super().__init__()

        if sensor_properties is None:
            raise Exception("No Sensor Properties Given")

        if sensor_life_days <= 0 or not isinstance(sensor_life_days, int):
            raise Exception("iCGM Sensor's sensor_life_days must be a positive non-zero integer")

        self.minutes_per_reading = 5
        self.num_readings_24hrs = int(24 * 60 / self.minutes_per_reading)  # hr * min/hr / min/reading

        self.current_sensor_bg = None
        self.current_sensor_bg_prediction = None

        self.sensor_life_days = sensor_life_days
        self.time_index = time_index
        self.current_datetime = current_datetime

        self.validate_time_index(self.time_index)

        self.reading_delay_buffer = []
        self.sensor_bg_history = []
        self.datetime_history = []

        self.initial_bias = sensor_properties["initial_bias"].values[0]
        self.phi_drift = sensor_properties["phi_drift"].values[0]
        self.bias_drift_range_start = sensor_properties["bias_drift_range_start"].values[0]
        self.bias_drift_range_end = sensor_properties["bias_drift_range_end"].values[0]
        self.bias_drift_oscillations = sensor_properties["bias_drift_oscillations"].values[0]
        self.bias_norm_factor = sensor_properties["bias_norm_factor"].values[0]
        self.noise_coefficient = sensor_properties["noise_coefficient"].values[0]
        self.delay_minutes = sensor_properties["delay"].values[0]
        self.random_seed = sensor_properties["random_seed"].values[0]
        self.bias_drift_type = sensor_properties["bias_drift_type"].values[0]

        self.calculate_sensor_bias_properties()

    def get_state(self):

        return SensorState(
            sensor_bg=self.current_sensor_bg,
            sensor_bg_prediction=self.current_sensor_bg_prediction
        )

    def validate_time_index(self, time_index):
        """Checks to see if the proposed sensor time index is within the sensor life"""
        before_sensor_starts = time_index < 0
        after_sensor_expires = time_index > (self.sensor_life_days * self.num_readings_24hrs - 1)  # time starts at 0
        if before_sensor_starts or after_sensor_expires:
            raise Exception("Sensor time_index {} outside of sensor life! ".format(str(time_index)))

    def store(self):
        """Store the current sensor state into history"""
        self.sensor_bg_history.append(self.current_sensor_bg)
        self.datetime_history.append(self.current_datetime)

    def update(self, next_datetime, **kwargs):
        """Step the sensor clock time forward"""

        if self.is_sensor_expired():
            raise SensorExpiredError("Sensor has expired.")

        true_bg = kwargs.get("patient_true_bg")
        true_bg_prediction = kwargs.get("patient_true_bg_prediction")

        self.current_sensor_bg = self.get_bg(true_bg)

        self.current_sensor_bg_prediction = None
        if true_bg_prediction is not None:
            self.current_sensor_bg_prediction = self.get_bg_trace(true_bg_prediction)

        self.store()
        self.time_index += 1
        self.current_datetime = next_datetime

    def is_sensor_expired(self):
        """
        Determine if sensor has expired.

        Returns
        -------
        bool
        """
        return self.time_index >= self.sensor_life_days * self.num_readings_24hrs

    def calculate_sensor_bias_properties(self):
        """Calculates the time series noise and bias drift properties based on other sensor properties"""

        num_days = 10  # CS 2020-06-20: Can this just be the sensor life or does that cause problems?

        # random seed for reproducibility
        np.random.seed(seed=self.random_seed)

        # noise component
        self.noise = np.random.normal(
            loc=0, scale=np.max([self.noise_coefficient, sys.float_info.epsilon]), size=self.num_readings_24hrs * num_days
        )

        # bias of individual sensor
        self.bias_factor = (self.bias_norm_factor + self.initial_bias) / (np.max([self.bias_norm_factor, 1]))

        if self.bias_drift_type == "random":

            # bias drift component over 10 days with cgm point every 5 minutes
            t = np.linspace(
                0, (self.bias_drift_oscillations * np.pi), self.num_readings_24hrs * num_days
            )  # this is the number of cgm points in 11 days
            sn = np.sin(t + self.phi_drift)

            self.drift_multiplier = np.interp(sn, (-1, 1), (self.bias_drift_range_start, self.bias_drift_range_start))

        if self.bias_drift_type == "linear":
            print("No 'linear' bias_drift_type implemented in iCGM Sensor")
            raise NotImplementedError

        if self.bias_drift_type == "none":
            self.drift_multiplier = np.ones(self.num_readings_24hrs * num_days)

    def get_bg(self, true_bg_value):
        """
        Calculate the iCGM value.

        If the sensor has a delay, the true_bg_value is stored and the delayed value from true_bg_history is used.

        Parameters
        ----------
        true_bg_value : float
            The true blood glucose value (mg/dL)
        time_index_offset : int
            The relative time offset from the current time_index (default of 0 uses the current time_index)
            Used when calculating future or past values without altering the time_index.

        Returns
        -------
        icgm_value : float
            The generated iCGM value

        """
        if true_bg_value is None:
            raise Exception("True bg must be a valid value, not None")

        # Get delayed true bg
        delayed_true_bg = np.nan
        self.reading_delay_buffer.append(true_bg_value)
        if len(self.reading_delay_buffer) > int(self.delay_minutes / self.minutes_per_reading):
            delayed_true_bg = self.reading_delay_buffer.pop(0)

        # Calculate value
        drift_multiplier = self.drift_multiplier[self.time_index]
        noise = self.noise[self.time_index]
        icgm_value = (delayed_true_bg * self.bias_factor * drift_multiplier) + noise

        return icgm_value

    def get_bg_trace(self, true_bg_trace):
        """
        This is STATELESS. Will compute the trace but not advance the state of the sensor. To advance
        sensor state use the update() function.

        Given a trace of true bg values, calculate the sensor bgs using the current sensor state

        Parameters
        ----------
        true_bg_trace : numpy float array
            The true blood glucose value trace (mg/dL)

        Returns
        -------
        sensor_bg_trace : numpy float array
            The array of iCGM sensor bgs generated from the true_bg_trace
        """
        delay_buffer_original = copy.copy(self.reading_delay_buffer)
        time_index_original = copy.copy(self.time_index)

        sensor_bg_trace = []

        for true_bg_value in true_bg_trace:
            sensor_bg = self.get_bg(true_bg_value)
            sensor_bg_trace.append(sensor_bg)
            self.time_index += 1

        # Reset any changed sensor state
        self.reading_delay_buffer = delay_buffer_original
        self.time_index = time_index_original

        return sensor_bg_trace

    def prefill_sensor_history(self, true_bg_history):
        """Prefills the sensor with true bgs and calculates the corresponding sensor bgs"""

        history_start_time = self.current_datetime - datetime.timedelta(minutes=len(true_bg_history) * 5)
        self.current_datetime = history_start_time

        try:
            for true_bg_value in true_bg_history:
                next_datetime = self.current_datetime + datetime.timedelta(minutes=5)
                self.update(next_datetime, patient_true_bg=true_bg_value)
        except SensorExpiredError as e:
            e_message = (
                "Trying to prefill past sensor life. "
                + "Establish the sensor at a different time_index or prefill with less data."
            )
            raise SensorExpiredError(e_message)

    def get_loop_inputs(self):
        """Get two arrays for dates and values, used for Loop input"""
        loop_bg_values = [max(40, min(400, round(bg))) for bg in self.sensor_bg_history]
        return self.datetime_history, loop_bg_values
