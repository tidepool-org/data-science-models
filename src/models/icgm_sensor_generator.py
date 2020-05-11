"""
Creates iCGM Sensors given a trueBG trace

The Dexcom G6 Specifications in this file are publicly available from:
    “EVALUATION OF AUTOMATIC CLASS III DESIGNATION FOR
    Dexcom G6 Continuous Glucose Monitoring System.” n.d.
    https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN170088.pdf.

"""

# %% Libraries
import pandas as pd
import numpy as np
import sys
from scipy.optimize import brute, fmin
import src.models.icgm_sensor_generator_functions as sf


# %% Definitions
class Sensor(object):
    """Base CGM Sensor Class"""

    def __init__(self):

        return

    def generate_value(self):

        raise NotImplementedError

    def generate_trace(self):

        raise NotImplementedError


class iCGMSensor(Sensor):
    """iCGM Sensor Object"""

    def __init__(self, sensor_properties, is_sample_sensor=False):

        super().__init__()

        if is_sample_sensor:
            self.initial_bias = 1.992889
            self.phi_drift = 2.158842
            self.bias_drift_range_start = 0.835931
            self.bias_drift_range_end = 1.040707
            self.bias_drift_oscillations = 1.041129
            self.bias_norm_factor = 55.000000
            self.noise_coefficient = 7.195753
            self.delay = 10
            self.random_seed = 0
            self.bias_drift_type = "random"
        else:
            self.initial_bias = sensor_properties["initial_bias"].values[0]
            self.phi_drift = sensor_properties["phi_drift"].values[0]
            self.bias_drift_range_start = sensor_properties["bias_drift_range_start"].values[0]
            self.bias_drift_range_end = sensor_properties["bias_drift_range_end"].values[0]
            self.bias_drift_oscillations = sensor_properties["bias_drift_oscillations"].values[0]
            self.bias_norm_factor = sensor_properties["bias_norm_factor"].values[0]
            self.noise_coefficient = sensor_properties["noise_coefficient"].values[0]
            self.delay = sensor_properties["delay"].values[0]
            self.random_seed = sensor_properties["random_seed"].values[0]
            self.bias_drift_type = sensor_properties["bias_drift_type"].values[0]

        self.is_sample_sensor = is_sample_sensor

        # random seed for reproducibility
        np.random.seed(seed=self.random_seed)

        # noise component
        self.noise = np.random.normal(
            loc=0, scale=np.max([self.noise_coefficient, sys.float_info.epsilon]), size=288 * 10
        )

        # bias of individual sensor
        self.bias_factor = (self.bias_norm_factor + self.initial_bias) / (np.max([self.bias_norm_factor, 1]))

        if self.bias_drift_type == "random":

            # bias drift component over 10 days with cgm point every 5 minutes
            t = np.linspace(
                0, (self.bias_drift_oscillations * np.pi), 288 * 10
            )  # this is the number of cgm points in 11 days
            sn = np.sin(t + self.phi_drift)

            self.drift_multiplier = np.interp(sn, (-1, 1), (self.bias_drift_range_start, self.bias_drift_range_start))

        if self.bias_drift_type == "linear":
            print("No 'linear' bias_drift_type implemented in iCGM Sensor")
            raise NotImplementedError

        if self.bias_drift_type == "none":
            self.drift_multiplier = np.ones(288 * 10)

        return

    def generate_value(self, true_bg_value, at_time):
        """
        This function returns the iCGM value of a true_bg_value

        WARNING: This function does not take into account time delay.
        If there is delay between the true BG and iCGM value, then pass in the true_bg_value from (at_time - delay).

        Parameters
        ----------
        true_bg_value : float
            The true blood glucose value (mg/dL)
        at_time : int
            The relative start time to the start of the sensor's drift and noise in 5-minute resolution
            (1 = 5min, 2 = 10min, etc)

        Returns
        -------
        icgm_value : float
            The generated iCGM value

        """

        icgm_value = ((true_bg_value * self.bias_factor) * self.drift_multiplier[at_time]) + self.noise[at_time]

        return icgm_value

    def generate_trace(self, true_bg_trace, start_time=0):
        """

        Parameters
        ----------
        true_bg_trace : numpy float array
            The true blood glucose value trace (mg/dL)
        start_time : int
            The relative start time to the start of the sensor's drift and noise in 5-minute resolution
            (1 = 5min, 2 = 10min, etc)

        Returns
        -------
        delayed_trace : numpy float array
            The iCGM array with an added front delay (if any)

        """

        end_time = start_time + len(true_bg_trace)
        drift = self.drift_multiplier[start_time:end_time]
        noise = self.noise[start_time:end_time]

        icgm_trace = ((true_bg_trace * self.bias_factor) * drift) + noise
        delayed_trace = np.insert(icgm_trace, 0, [np.nan] * int(self.delay / 5))

        return delayed_trace


class iCGMSensorGenerator(object):
    """iCGM Sensor Generator object which fits a Johnsonsu distribution to a true_bg_trace
    and generates sensors using this distribution"""

    def __init__(
        self,
        sc_thresholds=None,  # This is required only for iCGM sensors for now (A-G)
        sensor_batch_size=100,
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
        sensor_batch_size : int
            Number of sensors used when fitting the optimal distribution of sensor characteristics
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
        self.sensor_batch_size = sensor_batch_size
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

        if true_bg_trace is None:
            print("NO BG TRACE GIVEN! \n Creating 48 hour sinusoid dataset.")
            self.true_bg_trace = self.generate_test_trace()

        self.johnson_parameter_search_range, self.search_range_inputs = sf.get_search_range()

        # set the random seed for reproducibility
        np.random.seed(seed=random_seed)

        self.icgm_traces = None
        self.individual_sensor_properties = None
        self.batch_sensor_brute_search_results = None
        self.batch_sensor_properties = None

        return

    def generate_test_trace(self):
        """
        Creates a 48-hour sine as a test true_bg_trace dataset

        Returns
        -------
        true_bg_trace : numpy float array
            A test true_bg_trace
        """
        self.true_dataset_name = "48hours-sinusoid"
        true_df, true_df_inputs = sf.create_dataset(
            kind="sine",
            N=288 * 2,
            min_value=40,
            max_value=400,
            time_interval=5,
            flat_value=np.nan,
            oscillations=2,
            random_seed=self.random_seed,
        )
        true_bg_trace = np.array(true_df["value"])

        return true_bg_trace

    def fit(self):
        """Creates the batch of iCGM sensors fit to a true_bg_trace using brute force"""

        batch_sensor_brute_search_results = brute(
            sf.johnsonsu_icgm_sensor,
            self.johnson_parameter_search_range,
            args=(
                self.true_bg_trace,
                self.sc_thresholds,
                self.sensor_batch_size,
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

    def generate_sensors(self, n_sensors):

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
            sensors.append(iCGMSensor(sensor_properties=sensor_properties))

        self.n_sensors = n_sensors
        self.sensors = sensors  # Array of sensor objects

        return

    def calculate_result_tables(self):
        """Calculates the special controls results tables"""

        # using new (refactored) metrics
        df = sf.preprocess_data(self.true_bg_trace, self.icgm_traces, icgm_range=[40, 400], ysi_range=[0, 900])

        """ icgm special controls """
        icgm_special_controls_table = sf.calc_icgm_sc_table(df, "generic")

        """ new loss function """
        g6_loss, g6_table = sf.calc_dexcom_loss(df, self.n_sensors)
        if not self.use_g6_accuracy_in_loss:
            g6_loss = np.nan

        loss_score, percent_pass = sf.calc_icgm_special_controls_loss(icgm_special_controls_table, g6_loss)

        """ overall results """
        overall_metrics_table = sf.calc_overall_metrics(df)
        overall_metrics_table.loc["ICGM_PASS%", "icgmSensorResults"] = percent_pass
        overall_metrics_table.loc["LOSS_SCORE", "icgmSensorResults"] = loss_score

        # Get individual sensor special controls results
        trace_len = len(self.true_bg_trace)
        sensor_n_pairs = []
        sensor_icgm_sensor_results = []
        sensor_metrics = []
        for i in range(self.n_sensors):
            ind_sensor_df = df.iloc[trace_len * i : trace_len * (i + 1)]
            ind_sensor_special_controls_table = sf.calc_icgm_sc_table(ind_sensor_df, "generic")

            loss_score, percent_pass = sf.calc_icgm_special_controls_loss(ind_sensor_special_controls_table, g6_loss)

            sensor_metrics_table = sf.calc_overall_metrics(ind_sensor_df)
            sensor_metrics_table.loc["ICGM_PASS%", "icgmSensorResults"] = percent_pass
            sensor_metrics_table.loc["LOSS_SCORE", "icgmSensorResults"] = loss_score
            # sensor_metrics_table
            sensor_n_pairs.append(ind_sensor_special_controls_table["nPairs"].values)
            sensor_icgm_sensor_results.append(ind_sensor_special_controls_table["icgmSensorResults"].values)
            sensor_metrics.append(sensor_metrics_table.T)

        sensor_n_pair_cols = icgm_special_controls_table.T.add_suffix("_nPairs").columns

        sensor_results_cols = icgm_special_controls_table.T.add_suffix("_results").columns

        sensor_n_pairs = pd.DataFrame(sensor_n_pairs, columns=sensor_n_pair_cols)
        sensor_icgm_sensor_results = pd.DataFrame(sensor_icgm_sensor_results, columns=sensor_results_cols)
        ind_sensor_metrics = pd.concat(sensor_metrics).reset_index(drop=True)
        self.individual_sensor_properties.reset_index(drop=True, inplace=True)

        self.individual_sensor_properties = pd.concat(
            [self.individual_sensor_properties, ind_sensor_metrics, sensor_n_pairs, sensor_icgm_sensor_results], axis=1
        )

        dist_param_names = [
            "a",
            "b",
            "mu",
            "sigma",
            "batch_noise_coefficient",
            "bias_drift_range_min",
            "bias_drift_range_max",
            "batch_bias_drift_oscillations",
        ]

        dist_df = pd.DataFrame(self.dist_params, columns=["icgmSensorResults"], index=dist_param_names)

        dist_df.loc["bias_drift_type"] = self.bias_drift_type

        """ dexcom g6 accuracy metric (tables)"""
        # gsc = sf.calc_icgm_sc_table(df, "g6")
        # g1a = sf.calc_g6_table1A(df, n_sensors)
        # g1b = sf.calc_g6_table1BF(df, n_sensors, "B")
        # g1f = sf.calc_g6_table1BF(df, "F")
        # g3a = sf.calc_g6_table3AC(df, n_sensors, "A")
        # g3c = sf.calc_g6_table3AC(df, "C")
        # g4 = sf.calc_g6_table4(df, n_sensors)
        # g6 = sf.calc_g6_table6(df, n_sensors)

        input_settings_table = sf.capture_settings(
            self.sensor_batch_size,
            self.use_g6_accuracy_in_loss,
            self.bias_type,
            self.bias_drift_type,
            self.delay,
            self.random_seed,
        )

        input_names = [
            "TRUE.kind",
            "TRUE.N",
            "TRUE.min_value",
            "TRUE.max_value",
            "TRUE.time_interval",
        ]

        true_df_inputs = pd.DataFrame(
            [
                self.true_dataset_name,
                len(self.true_bg_trace),
                np.min(self.true_bg_trace),
                np.max(self.true_bg_trace),
                5,
            ],
            columns=["icgmSensorResults"],
            index=input_names,
        )

        results_df = pd.concat(
            [input_settings_table, true_df_inputs, self.search_range_inputs, dist_df, overall_metrics_table, g6_table,],
            sort=False,
        )

        batch_sensor_properties = results_df[~results_df.index.duplicated(keep="first")]
        sc_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        batch_sensor_properties.drop(index=sc_letters, inplace=True)
        batch_sensor_properties.drop(columns=["dexG6"], inplace=True)

        batch_sc_table = icgm_special_controls_table

        batch_sc_npairs = pd.DataFrame(batch_sc_table["nPairs"].T.add_suffix("_nPairs"))
        batch_sc_npairs.columns = ["icgmSensorResults"]

        batch_sc_results = pd.DataFrame(batch_sc_table["icgmSensorResults"].T.add_suffix("_results"))

        self.batch_sensor_properties = pd.concat([batch_sensor_properties, batch_sc_npairs, batch_sc_results])

        return

    def get_batch_results(self):
        """
        Gets the batch sensor generator results

        Returns
        -------
        batch_results : tuple
            The generated sensor icgm_traces and the individual & batch properties
        """

        batch_results = (self.icgm_traces, self.individual_sensor_properties, self.batch_sensor_properties)

        return batch_results
