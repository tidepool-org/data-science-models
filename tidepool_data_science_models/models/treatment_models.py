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

    def run(self, t, treatment_amount, **kwargs):
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
        self._theta = kwargs.get("theta", 20)
        self._Kcl = kwargs.get("kcl", 1)

        self.absorption_time_hrs_tau_map = {
            25: 1,
            30: 2,
            35: 3,
            40: 4,
            45: 5,
            50: 6,
            55: 7,
            60: 8,
            65: 9,
            70: 10,
            75: 11,
            80: 13,
            85: 14,
            90: 15,
            95: 16,
            100: 17,
            105: 18,
            110: 19,
            115: 20,
            120: 21,
            125: 22,
            130: 23,
            135: 24,
            140: 26,
            145: 27,
            150: 28,
            155: 29,
            160: 30,
            165: 31,
            170: 32,
            175: 33,
            180: 34,
            185: 35,
            190: 36,
            195: 38,
            200: 39,
            205: 40,
            210: 41,
            215: 42,
            220: 43,
            225: 44,
            230: 45,
            235: 46,
            240: 47,
            245: 48,
            250: 49,
            255: 51,
            260: 52,
            265: 53,
            270: 54,
            275: 55,
            280: 56,
            285: 57,
            290: 58,
            295: 59,
            300: 60,
            305: 61,
            310: 62,
            315: 64,
            320: 65,
            325: 66,
            330: 67,
            335: 68,
            340: 69,
            345: 70,
            350: 71,
            355: 72,
            360: 73,
            365: 74,
            370: 76,
            375: 77,
            380: 78,
            385: 79,
            390: 80,
            395: 81,
            400: 82,
            405: 83,
            410: 84,
            415: 85,
            420: 86,
            425: 87,
            430: 89,
            435: 90,
            440: 91,
            445: 92,
            450: 93,
            455: 94,
            460: 95,
            465: 96,
            470: 97,
            475: 98,
            480: 99,
            485: 100,
            490: 102,
            495: 103,
            500: 104,
            505: 105,
            510: 106,
            515: 107,
            520: 108,
            525: 109,
            530: 110,
            535: 111,
            540: 112,
            545: 114,
            550: 115,
            555: 116,
            560: 117,
            565: 118,
            570: 119,
            575: 120,
            580: 121,
            585: 122,
            590: 123,
            595: 124,
        }

    def run(self, num_hours, carb_amount, carb_absorb_minutes, five_min=True):
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

        if carb_absorb_minutes < 25 or carb_absorb_minutes > 595:  # 25 minutes or ~10 hours
            raise Exception("Absorption time minutes must be ")

        minutes_increment_5 = 5 * np.floor(carb_absorb_minutes / 5)  # round to nearest five
        tau = self.absorption_time_hrs_tau_map[minutes_increment_5]
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


class LoopInsulinModel(TreatmentModel):
    def __init__(self):
        super().__init__("Loop_vX.X")
        raise NotImplementedError


class LoopCarbModel(TreatmentModel):
    def __init__(self):
        super().__init__("Loop_vX.X")
        raise NotImplementedError
