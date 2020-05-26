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

    def run(self, carb_amount, insulin_amount=np.nan, num_hours=8):
        """
        Compute a num_hours long, 5-min interval time series metabolic response to insulin and carbs inputs
        at t0. Carbs and insulin can be either zero or non-zero.

        If insulin is not given, the amount is automatically determined by the carb input using bolus
        wizard logic.

        Parameters
        ----------
        carb_amount: float
            Amount of carbs
        insulin_amount

        Returns
        -------

        """
        if num_hours < 0:
            raise ValueError("Number of hours for simulation can't be negative.")

        if num_hours > 24:
            raise ValueError("Number of hours for simulation can't be more than 24.")

        # create a time series
        t = get_timeseries(num_hours, five_min=False)
        t_5min = get_timeseries(num_hours, five_min=True)

        # if insulin amount is not given,
        # calculate carb amount like a bolus calculator
        if np.isnan(insulin_amount):
            insulin_amount = carb_amount / self._cir  # insulin amount

        # insulin model
        if insulin_amount != 0:

            # model constants
            i_t, iob_t = self.insulin_model.run(t, insulin_amount=insulin_amount)

            ie_5min = i_t[t_5min]
            iob_5min = iob_t[t_5min]
            decrease_due_to_insulin = np.append(0, ie_5min[1:] - ie_5min[:-1])

        else:
            decrease_due_to_insulin = np.zeros(len(t_5min))
            iob_5min = np.zeros(len(t_5min))

        # carb model
        if carb_amount > 0:

            c_t = self.carb_model.run(t, carb_amount=carb_amount)
            ce_5min = c_t[t_5min]
            increase_due_to_carbs = np.append(0, ce_5min[1:] - ce_5min[:-1])

        else:
            increase_due_to_carbs = np.zeros(len(t_5min))

        net_change_in_bg = decrease_due_to_insulin + increase_due_to_carbs

        # +CS - Why are we returning the carb and insulin amt?
        return net_change_in_bg, t_5min, carb_amount, insulin_amount, iob_5min

    def get_iob_from_sbr(self, sbr_actual):
        """
        Compute insulin on board every 5 minutes for 8 hours following the initial condition
        being insulin on board from the scheduled basal rate for 8 hours.

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
        # Cameron added explanation since it was unclear what was going on until I stared
        # at it for a while. Ed, please edit if these aren't correct.

        # Step 1: Get 8 hr iob from a bolus that is 1/12 of the scheduled basal rate.
        #         This assumes basal rate is a series of boluses at 5 min intervals.
        # TODO: optionally expose these as arguments to support other pumps or insulin curves (e.g. might need more
        #  time to reach steady state or full decay with some curves), but need sufficient testing
        num_hours_pre_t0 = 8
        num_hours_post_t0 = 8
        minutes_per_pump_pulse = 5

        # E.g. 1 pulse every 5 minutes: pulses/hr = 60 min/hr / 5 min/pulse
        num_basal_pulses_per_hour = int(MINUTES_PER_HOUR / minutes_per_pump_pulse)

        # TODO: CS - this is unrealistic pump behavior since pumps deliver in increments of U/pulse
        basal_amount_per_5min = (
            sbr_actual / num_basal_pulses_per_hour
        )  # U/pulse = U/hr / pulse/hr

        t = get_timeseries(num_hours=num_hours_pre_t0, five_min=False)
        t_5min = get_timeseries(num_hours=num_hours_pre_t0, five_min=True)
        i_t, iob_t = self.insulin_model.run(t, insulin_amount=basal_amount_per_5min)
        iob_t5min = iob_t[t_5min]

        # Step 2: Add hours post t0 to simulation
        iob_with_zeros = np.append(
            iob_t5min, np.zeros(num_hours_post_t0 * num_basal_pulses_per_hour)
        )

        # Step 3: Copy the activity curves across the whole matrix
        iob_matrix = np.tile(
            iob_with_zeros, (num_hours_post_t0 * num_basal_pulses_per_hour, 1)
        ).T

        # Step 4: Shift each activity curve to mimic the basal pulse at each time step
        nrows, ncols = np.shape(iob_matrix)
        for t_pre in np.arange(1, ncols):
            iob_matrix[:, t_pre] = np.roll(iob_matrix[:, t_pre], t_pre)

        # Step 5: Fill the upper triangle with zeros
        #         FIXME: CS - is this necessary? My numpy roll() above seems have zeros already
        iob_matrix_tri = iob_matrix * np.tri(nrows, ncols, 0)

        # Step 6: Sum across the curves to get the iob at every time step
        iob_sbr_t = np.sum(iob_matrix_tri, axis=1)

        # Step 7: Just get the last 8 hours
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
