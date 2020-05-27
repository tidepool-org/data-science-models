"""
This file houses everything related to the insulin and carb modeling math.
"""

import numpy as np

from tidepool_data_science_models.utils import MINUTES_PER_HOUR, STEADY_STATE_IOB_FACTOR_FDA, get_timeseries
from tidepool_data_science_models.models.treatment_models import PalermInsulinModel, CesconCarbModel


class SimpleMetabolismModel(object):
    """
    A class with modular ability to run different insulin and carb algorithms
    for modeling metabolism of insulin and carbs in the body. The 
    """

    def __init__(
        self,
        insulin_sensitivity_factor,
        carb_insulin_ratio,
        insulin_model_name="palerm",
        carb_model_name="cescon",
    ):
        """
        Parameters
        ----------
        insulin_sensitivity_factor: float
            How many mg/dL are reduced by 1 unit of insulin, units: mg/dL / U

        carb_insulin_ratio: float
            How many g carbs are offset by 1 unit of insulin, units: g / mg/dL

        insulin_model_name: str
            Name of the insulin model to use

        carb_model_name: str
            Name of the carb model to use
        """
        self._cir = carb_insulin_ratio
        self._isf = insulin_sensitivity_factor

        if insulin_model_name == "palerm":
            self.insulin_model = PalermInsulinModel(
                isf=insulin_sensitivity_factor, cir=carb_insulin_ratio
            )
        else:
            raise ValueError(
                "{} not a recognized insulin model".format(insulin_model_name)
            )

        if carb_model_name == "cescon":
            self.carb_model = CesconCarbModel(
                isf=insulin_sensitivity_factor, cir=carb_insulin_ratio
            )
        else:
            raise ValueError("{} not a recognized carb model.".format(carb_model_name))

    def run(self, carb_amount, insulin_amount=np.nan, num_hours=8, five_min=True):
        """
        Compute a num_hours long, 5-min interval time series metabolic response to insulin and carbs inputs
        at t0. Carbs and insulin can be either zero or non-zero.

        If insulin is not given, the amount is automatically determined by the carb input using bolus
        wizard logic.

        Parameters
        ----------
        carb_amount: float
            Amount of carbs

        insulin_amount: float
            Amount of insulin, if not given is calculated based on carb_amount

        num_hours: float
            Number of hours to run the simulation past t0

        five_min: bool
            Where to use 5 minute subsampling, if False default is 1 minute

        Returns
        -------
        (np.array, np.array, float, np.array)
            combined_delta_bg - The delta bg as a result of input insulin and carbs
            t_min - time series that matches the simulation outputs
            insulin_amount - Input insulin or insulin computed from carbs if np.nan is passed in
            iob - The insulin on board
        """
        if num_hours < 0:
            raise ValueError("Number of hours for simulation can't be negative.")

        if num_hours > 24:
            raise ValueError("Number of hours for simulation can't be more than 24.")

        if carb_amount < 0:  # Note: insulin can be negative
            raise ValueError("Carbs must be greater than zero.")
        
        # if insulin amount is not given,
        # calculate carb amount like a bolus calculator
        if np.isnan(insulin_amount):
            insulin_amount = carb_amount / self._cir  # insulin amount

        sim_data_len = num_hours * MINUTES_PER_HOUR
        if five_min:
            sim_data_len = int(sim_data_len / 5)

        # Init arrays to return
        combined_delta_bg = np.zeros(sim_data_len)
        iob = np.zeros(sim_data_len)

        # insulin model
        if insulin_amount > 0:
            t_min, bg_delta_insulin, bg, iob = self.insulin_model.run(
                num_hours, insulin_amount=insulin_amount, five_min=five_min
            )
            combined_delta_bg += bg_delta_insulin

        # carb model
        if carb_amount > 0:
            t_min, bg_delta_carb, bg = self.carb_model.run(
                num_hours, carb_amount=carb_amount, five_min=five_min
            )
            combined_delta_bg += bg_delta_carb

        # +CS - Why are we returning the carb and insulin amt?
        return combined_delta_bg, t_min, insulin_amount, iob

    def get_iob_from_sbr(self, sbr_actual):
        """
        Compute insulin on board due to the assumption that the schedule basal rate (sbr)
        has been active for at least N_pre hours (8 hours here) prior to the start of the simulation.
        The effect of the insulin due to the sbr prior to the start of the simulation results in
        an initial amount of insulin on board every 5 minutes over the following N_post hours (8 hours again).

        Parameters
        ----------
        sbr_actual: float
            The scheduled basal rate

        isf: float
            The insulin sensitivity factor

        cir: float
            The carb to insulin ratio

        Returns
        -------
        np.array
            The insulin on board every five munutes for 8 hours
        """
        # Step 1: Get 8 hr iob from a bolus that is 1/12 of the scheduled basal rate.
        #         This assumes basal rate is a series of boluses at 5 min intervals.
        # TODO: optionally expose these as arguments to support other pumps or insulin curves (e.g. might need more
        #  time to reach steady state or full decay with some curves), but need sufficient testing
        num_hours_pre_t0 = 8
        num_hours_post_t0 = 8
        minutes_per_pump_pulse = 5

        # E.g. 1 pulse every 5 minutes: 12 pulses/hr = 60 min/hr / 5 min/pulse
        num_basal_pulses_per_hour = int(MINUTES_PER_HOUR / minutes_per_pump_pulse)

        # NOTE: This is unrealistic pump behavior since pumps deliver in increments of U/pulse,
        #       but in the steady state case it has an insignificant effect on the result.
        basal_amount_per_5min = (
            sbr_actual / num_basal_pulses_per_hour
        )  # U/pulse = U/hr / pulse/hr

        t, delta_bg_t, bg_t, iob_t = self.insulin_model.run(
            num_hours_pre_t0, insulin_amount=basal_amount_per_5min, five_min=True
        )

        # Step 2: Add hours post t0 to simulation
        iob_with_zeros = np.append(
            iob_t, np.zeros(num_hours_post_t0 * num_basal_pulses_per_hour)
        )

        # Step 3: Copy the activity curves across the whole matrix
        iob_matrix = np.tile(
            iob_with_zeros, (num_hours_post_t0 * num_basal_pulses_per_hour, 1)
        ).T

        # Step 4: Shift each activity curve to mimic the basal pulse at each time step
        nrows, ncols = np.shape(iob_matrix)
        for t_pre in np.arange(1, ncols):
            iob_matrix[:, t_pre] = np.roll(iob_matrix[:, t_pre], t_pre)

        # TODO: In theory we shouldn't need this, but until this function is well tested
        #       with different values of num_hours_pre_t0 and num_hours_post_t0 leaving it as
        #       a safeguard.
        if iob_matrix[0, -1] != 0.0:
            raise ValueError(
                "Algorithm expects zeros in the upper right triangle. Please review."
            )

        # Step 5: Sum across the curves to get the iob at every time step
        iob_sbr_t = np.sum(iob_matrix, axis=1)

        # Step 6: Just get the last 8 hours
        slice_index = int(
            (num_hours_post_t0 * MINUTES_PER_HOUR / minutes_per_pump_pulse) - 1
        )
        iob_sbr_t = iob_sbr_t[slice_index:-1]

        return iob_sbr_t

    def get_steady_state_iob_from_sbr(self, sbr, use_fda_submission_constant=True):
        """
        Get the scalar amount of insulin that is the steady state insulin on board
        for a given scheduled basal rate.

        Parameters
        ----------
        sbr: float
            The scheduled basal rate

        use_fda_submission_constant: bool
            Whether to use the constant multiplier determined from the FDA risk analysis
            or to compute it from the internal insulin curves.

        Returns
        -------
        float
            The steady state insulin
        """

        if use_fda_submission_constant:
            steady_state_iob = sbr * STEADY_STATE_IOB_FACTOR_FDA
        else:
            iob_t_sbr_activity = self.get_iob_from_sbr(sbr)
            steady_state_iob = iob_t_sbr_activity[0]

        return steady_state_iob
