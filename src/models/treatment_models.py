"""
This file houses the various insulin and carb activity curve models
"""

import numpy as np


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

    def run(self, t, insulin_amount):
        """
        Parameters
        ----------
        t: np.array
            The time input for computing the activity curve.

        insulin_amount: float
            The amount of insulin to use for running the model
        """
        ISF = self._isf

        tau1 = self._tau1
        tau2 = self._tau2
        Kcl = self._Kcl

        insulin_equation = (
            insulin_amount
            * (1 / (Kcl * (tau2 - tau1)))
            * (np.exp(-t / tau2) - np.exp(-t / tau1))
        )
        ia_t = np.cumsum(insulin_equation)
        iob_t = insulin_amount - ia_t

        i_t = -ISF * ia_t

        return i_t, iob_t


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

    def run(self, t, carb_amount):
        """
        Parameters
        ----------
        t: np.array
            The time input for computing the activity curve.

        carb_amount: float
            The amount of carbs to use for running the model
        """
        K = self._isf / self._cir  # carb gain
        tau = self._tau
        theta = self._theta
        c_t = (
            K
            * carb_amount
            * (1 - np.exp((theta - t) / tau))
            * np.heaviside(t - theta, 1)
        )

        # TODO: return cob here as well since insulin does it and it's more efficient to while here
        return c_t


class LoopInsulinModel(TreatmentModel):
    def __init__(self):
        raise NotImplementedError


class LoopCarbModel(TreatmentModel):
    def __init__(self):
        raise NotImplementedError
