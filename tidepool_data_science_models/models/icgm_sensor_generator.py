"""
Creates iCGM Sensors given a trueBG trace

The Dexcom G6 Specifications in this file are publicly available from:
    “EVALUATION OF AUTOMATIC CLASS III DESIGNATION FOR
    Dexcom G6 Continuous Glucose Monitoring System.” n.d.
    https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN170088.pdf.

"""

# %% Libraries
import numpy as np
from scipy.optimize import brute, fmin
from tidepool_data_science_models.models.icgm_sensor import iCGMSensor
import tidepool_data_science_models.models.icgm_sensor_generator_functions as sf
import multiprocessing
multiprocessing.set_start_method("fork")


# %% Definitions
class iCGMSensorGenerator(object):
    """iCGM Sensor Generator object which fits a Johnsonsu distribution to a true_bg_trace
    and generates sensors using this distribution"""

    def __init__(
        self,
        sc_thresholds=None,  # This is required only for iCGM sensors for now (A-G)
        batch_training_size=30,
        use_g6_accuracy_in_loss=False,
        bias_type="percentage_of_value",
        bias_drift_type="random",
        random_seed=0,
        verbose=False,
        true_bg_trace=None,
        true_dataset_name="default",
    ):
        """
        Sensor Generator Initialization

        Parameters
        ----------
        sc_thresholds : float array
            The 7 special control thresholds A-G
        use_g6_accuracy_in_loss : bool
            Whether or not to use the G6 accuracy loss during  fit
        bias_type : str
            Type of overall bias used which defines the normalization factor
        bias_drift_type : str
            Type of drift used in the sensor bias (random, linear, none)
        random_seed : int
            Random seed used throughout generator for reproducible sensors and values
        verbose : bool
            Verbosity setting for the brute force distribution parameter search
        true_bg_trace : float array
            The time-series of true bgs the iCGM distribution is fit to
        true_dataset_name : str
            Name of the true bg dataset used to fit
        """

        if sc_thresholds is None:
            sc_thresholds = [
                0.85,
                0.70,
                0.80,
                0.98,
                0.99,
                0.99,
                0.87,
            ]  # This is required only for iCGM sensors (Criteria A-G)

        self.sc_thresholds = sc_thresholds
        self.batch_training_size = batch_training_size
        self.use_g6_accuracy_in_loss = use_g6_accuracy_in_loss
        self.bias_type = bias_type
        self.bias_drift_type = bias_drift_type
        self.random_seed = random_seed
        self.verbose = verbose
        self.true_bg_trace = true_bg_trace
        self.true_dataset_name = true_dataset_name

        # pick delay based upon data in:
        # Vettoretti et al., 2019, Sensors 2019, 19, 5320
        if use_g6_accuracy_in_loss:
            self.delay = 5  # time delay between iCGM value and true value
        else:
            self.delay = 10

        self.johnson_parameter_search_range, self.search_range_inputs = sf.get_search_range()

        # set the random seed for reproducibility
        np.random.seed(seed=random_seed)

        self.icgm_traces = None
        self.individual_sensor_properties = None
        self.batch_sensor_brute_search_results = None
        self.batch_sensor_properties = None
        self.dist_params = None

        return

    def fit(self, true_bg_trace=None):
        """Fits the optimal sensor characteristics fit to a true_bg_trace using a brute search range

        Parameters
        ----------
        true_bg_trace : float array
            The true_bg_trace (mg/dL) used to fit a johnsonsu distribution
        training_size : int
            Number of sensors used when fitting the optimal distribution of sensor characteristics

        """

        if true_bg_trace is None:
            raise Exception("No true_bg_trace given")

        self.true_bg_trace = true_bg_trace

        batch_sensor_brute_search_results = brute(
            sf.johnsonsu_icgm_sensor,
            self.johnson_parameter_search_range,
            args=(
                true_bg_trace,
                self.sc_thresholds,
                self.batch_training_size,
                self.bias_type,
                self.bias_drift_type,
                self.delay,
                self.random_seed,
                self.verbose,
                self.use_g6_accuracy_in_loss,
            ),
            workers=-1,
            full_output=True,
            finish=fmin,  # fmin will look for a local minimum around the grid point
        )

        self.batch_sensor_brute_search_results = batch_sensor_brute_search_results
        self.dist_params = self.batch_sensor_brute_search_results[0]

        return

    def generate_sensors(self, n_sensors, sensor_start_datetime, sensor_start_time_index=0):

        if self.dist_params is None:
            raise Exception("iCGM Sensor Generator has not been fit() to a true_bg_trace distribution.")

        (
            a,
            b,
            mu,
            sigma,
            noise_coefficient,
            bias_drift_range_min,
            bias_drift_range_max,
            bias_drift_oscillations,
        ) = self.dist_params

        bias_drift_range = [bias_drift_range_min, bias_drift_range_max]

        # STEP 3 apply the results
        # Convert to a generate_sensor(global_params) --> Sensor(obj)
        self.icgm_traces, self.individual_sensor_properties = sf.generate_icgm_sensors(
            self.true_bg_trace,
            dist_params=self.dist_params[:4],
            n_sensors=n_sensors,
            bias_type=self.bias_type,
            bias_drift_type=self.bias_drift_type,
            bias_drift_range=bias_drift_range,
            bias_drift_oscillations=bias_drift_oscillations,
            noise_coefficient=noise_coefficient,
            delay=self.delay,
            random_seed=self.random_seed,
        )

        sensors = []

        for sensor_num in range(n_sensors):
            sensor_properties = self.individual_sensor_properties.loc[sensor_num]
            sensors.append(
                iCGMSensor(
                    sensor_properties=sensor_properties,
                    time_index=sensor_start_time_index,
                    current_datetime=sensor_start_datetime,
                )
            )

        self.n_sensors = n_sensors
        self.sensors = sensors  # Array of sensor objects

        return sensors
