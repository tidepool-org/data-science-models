"""
This file houses the various insulin and carb activity curve models
"""

import numpy as np

from tidepool_data_science_models.utils import get_timeseries


class TreatmentModel(object):
    """
    Base class for treatments, e.g. insulin, carbs
    """

    def __init__(self, name):
        self.name = name

    def run(self, t, treatment_amount):
        raise NotImplementedError

    def __repr__(self):
        return "{} Model".format(self.name)

    def __str__(self):
        return "{} Model".format(self.name)

    def get_name(self):
        return self.name


class PalermInsulinModel(TreatmentModel):
    """
    Insulin model chosen for risk analysis of Loop for FDA submission. Justification and details:

    https://colab.research.google.com/drive/17_0OTtM3stNUIyUWZGyB011x8yfvuis5#forceEdit=true&sandboxMode=true&scrollTo=7dJX4WMI205n
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            Arguments specific to the model.
            Requires insulin sensitivity factor (isf) and and carb insulin ratio (cir)
        """
        super().__init__("Palerm")
        self._isf = kwargs["isf"]
        self._cir = kwargs["cir"]

        # Defaults are from Palerm paper
        self._tau1 = kwargs.get("tau1", 55)
        self._tau2 = kwargs.get("tau2", 70)
        self._Kcl = kwargs.get("kcl", 1)

    def run(self, num_hours, insulin_amount, five_min=True):
        """
        Run the model for num hours assuming that the insulin amount
        is given at t=0.

        Parameters
        ----------
        num_hours: float
            How long to compute the effect

        insulin_amount: float
            The amount of insulin to use for running the model

        five_min: bool
            If true, run the model in increments of 5 minutes, otherwise
            1 minute

        Returns
        -------
        (np.array, np.array, np.array, np.array)
            t: The time series in minutes
            bg_delta: The change in bg for each time in t
            bg: The bg for each time in t starting at 0
            iob: The insulin on board for each time in t
        """
        t_min = get_timeseries(num_hours, five_min=False)

        isf = self._isf

        tau1 = self._tau1
        tau2 = self._tau2
        kcl = self._Kcl

        insulin = (
            insulin_amount
            * (1 / (kcl * (tau2 - tau1)))
            * (np.exp(-t_min / tau2) - np.exp(-t_min / tau1))
        )

        insulin_cleared = np.cumsum(insulin)
        iob = insulin_amount - insulin_cleared
        bg = -1 * isf * insulin_cleared

        # Optionally subsample
        if five_min:
            t_min = get_timeseries(num_hours, five_min=True)
            bg = bg[t_min]
            iob = iob[t_min]

        bg_delta = np.append(0, bg[1:] - bg[:-1])

        return t_min, bg_delta, bg, iob


class CesconCarbModel(TreatmentModel):
    """
    Carb model chosen for risk analysis of Tidepool Loop and FDA submission. Justification:

    https://colab.research.google.com/drive/17_0OTtM3stNUIyUWZGyB011x8yfvuis5#forceEdit=true&sandboxMode=true&scrollTo=7dJX4WMI205n
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            Arguments specific to the model.
            Requires insulin sensitivity factor (isf) and and carb insulin ratio (cir)
        """
        super().__init__("Cescon")
        self._isf = kwargs["isf"]
        self._cir = kwargs["cir"]

        # Defaults are from Cescon paper
        self._tau = kwargs.get("tau", 42)
        self._theta = kwargs.get("theta", 20)
        self._Kcl = kwargs.get("kcl", 1)

    def run(self, num_hours, carb_amount, five_min=True):
        """
        Run the model for num hours assuming that the carb amount
        is given at t=0.

        Parameters
        ----------
        num_hours: float
            The amount of time in hours to compute the effect

        carb_amount: float
            The amount of carbs to use for running the model

        five_min: bool
            If true, run the model in increments of 5 minutes, otherwise
            1 minute

        Returns
        -------
        (np.array, np.array, np.array)
            t: The time series in minutes
            bg_delta: The change in bg for each time in t
            bg: The bg for each time in t starting at 0
        """
        t_min = get_timeseries(num_hours, five_min=False)

        if five_min:
            t_min = get_timeseries(num_hours, five_min=True)

        K = self._isf / self._cir  # mg/dL / g = (mg/dL / U) / (g / U)
        tau = self._tau
        theta = self._theta

        # mg/dL * min = (mg/dL / g) * g * min
        bg = (
            K
            * carb_amount
            * (1 - np.exp((theta - t_min) / tau))
            * np.heaviside(t_min - theta, 1)
        )

        # mg/dL / min
        bg_delta = np.append(0, bg[1:] - bg[:-1])

        return t_min, bg_delta, bg

class Type2InsulinModel(TreatmentModel):

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            Arguments specific to the model.
            Requires insulin sensitivity factor (isf) and and carb insulin ratio (cir)
        """
        super().__init__("T2Insulin")
        self._isf = kwargs["isf"]
  
        # Defaults are averaged from the Hovorka paper
        self._gsf = kwargs.get("gsf", 0)
        self._bbg = kwargs.get("bbg", 100)
        self._ipr = kwargs.get("ipr", 0)

    def run(self, num_hours, blood_glucose, five_min=True):
        """
        Run the model for num hours assuming that the carb amount
        is given at t=0.

        Parameters
        ----------
        num_hours: float
            The amount of time in hours to compute the effect

        carb_amount: float
            The amount of carbs to use for running the model

        five_min: bool
            If true, run the model in increments of 5 minutes, otherwise
            1 minute

        Returns
        -------
        (np.array, np.array, np.array)
            t: The time series in minutes
            bg_delta: The change in bg for each time in t
            bg: The bg for each time in t starting at 0
        """
    
        isf = self._isf
        
        glucose_sensitivity_factor = self._gsf
        basal_blood_glucose = self._bbg
        insulin_production_rate = self._ipr
        
        t_min = get_timeseries(num_hours, five_min=five_min)
        bg_delta = np.zeros_like(t_min, dtype=float) 
        ei = np.zeros_like(t_min, dtype=float) 

        bg_above_baseline = np.max((blood_glucose - basal_blood_glucose, 0))
        post_hepatic_insulin = np.max(glucose_sensitivity_factor * bg_above_baseline, 0) + insulin_production_rate * basal_blood_glucose

        if five_min:
            post_hepatic_insulin *= 5
        
        insulin_effect = -1 * isf * post_hepatic_insulin

        bg_delta[1] = insulin_effect
        ei[1] = post_hepatic_insulin

        return t_min, bg_delta, ei

        
class LoopInsulinModel(TreatmentModel):
    def __init__(self):
        super().__init__("Loop_vX.X")
        raise NotImplementedError


class LoopCarbModel(TreatmentModel):
    def __init__(self):
        super().__init__("Loop_vX.X")
        raise NotImplementedError
