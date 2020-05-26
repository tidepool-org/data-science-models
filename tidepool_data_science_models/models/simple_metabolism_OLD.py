"""
These are the original functions for modeling insulin and carbs for the FDA loop risk analysis
 written by Ed (with superficial modifications). They are here for reference and testing.
"""


import numpy as np

from tidepool_data_science_models.models.simple_metabolism_model import STEADY_STATE_IOB_FACTOR_FDA


def simple_metabolism_model(
    carb_amount=0,  # grams (g)
    insulin_amount=np.nan,  # units of insulin (U)
    cir=12.5,  # carb-to-insulin-ratio (g/U)
    isf=50,  # insulin sensitivity factor (mg/dL/U)
):
    """
    Compute an 8 hour long, 5-min interval time series metabolic response to insulin and carbs inputs
    at t0. If insulin is not given,

    Args:
        carb_amount: carb amount at t0 (g)
        insulin_amount: insulin amount at t0 (U)
        cir: carb to insulin ratio (g/U)
        isf: insulin sensitivity factor (mg/dL/U)

    Returns:
        tuple: (
            net_change_in_bg,
            t_5min,
            carb_amount,
            insulin_amount,
            iob_5min
            )
    """
    # +CS
    minutes_in_model = 8 * 60  # 8 hr * 60 minutes/hr

    # +CS why do we have 2 time series? Reduce computation with just 5 min time series?

    # create a time series
    t = np.arange(0, minutes_in_model, 1)  # in minutes
    t_5min = np.arange(0, minutes_in_model, 5)

    # +CS Why do we assume the an insulin amount if it's not given?
    # This could be more generalized?

    # if insulin amount is not given,
    # calculate carb amount like a bolus calculator
    if np.isnan(insulin_amount):
        insulin_amount = carb_amount / cir  # insulin amount

    # insulin model
    if insulin_amount != 0:

        # model constants
        tau1 = 55
        tau2 = 70
        Kcl = 1

        insulin_equation = (
            insulin_amount
            * (1 / (Kcl * (tau2 - tau1)))
            * (np.exp(-t / tau2) - np.exp(-t / tau1))
        )

        ia = np.cumsum(insulin_equation)
        iob = insulin_amount - ia
        iob_5min = iob[t_5min]
        insulin_effect = -isf * ia
        ie_5min = insulin_effect[t_5min]
        decrease_due_to_insulin_5min = np.append(0, ie_5min[1:] - ie_5min[:-1])

    else:
        decrease_due_to_insulin_5min = np.zeros(len(t_5min))
        iob_5min = np.zeros(len(t_5min))

    # carb model
    if carb_amount > 0:
        K = isf / cir  # carb gain
        tau = 42
        theta = 20
        c_t = (
            K
            * carb_amount
            * (1 - np.exp((theta - t) / tau))
            * np.heaviside(t - theta, 1)
        )
        ce_5min = c_t[t_5min]
        increase_due_to_carbs_5min = np.append(0, ce_5min[1:] - ce_5min[:-1])

    else:
        increase_due_to_carbs_5min = np.zeros(len(t_5min))

    net_change_in_bg_5min = decrease_due_to_insulin_5min + increase_due_to_carbs_5min

    # +CS - Why are we returning the carb and insulin amt?
    return net_change_in_bg_5min, t_5min, carb_amount, insulin_amount, iob_5min


def get_iob_from_sbr(sbr_actual):
    """
    Compute insulin on board for 8 hours following with the initial condition
    being insulin on board from the scheduled basal rate for 8 hours.

    Parameters
    ----------
    sbr_actual
    isf
    cir

    Returns
    -------

    """
    # TODO: Further clarify this
    # Cameron added explanation since it was unclear what was going on until I stared
    # at it for a while. Ed, please edit if these aren't correct.

    # Step 1: Get 8 hr iob from a bolus that is 1/12 of the scheduled basal rate.
    #         This assumes basal rate is a series of boluses at 5 min intervals.
    _, _, _, _, iob_sbr = simple_metabolism_model(
        carb_amount=0,
        insulin_amount=sbr_actual / 12,
        cir=0,  # This doesn't matter for this use of the model
        isf=0,  # Same as above
    )

    # Step 2: Allocate
    iob_with_zeros = np.append(iob_sbr, np.zeros(8 * 12))

    # Step 3: Copy the decay curves across the whole matrix
    iob_matrix = np.tile(iob_with_zeros, (8 * 12, 1)).T

    # Step 4: Shift each decay curve by the number of time steps
    nrows, ncols = np.shape(iob_matrix)
    for t_pre in np.arange(1, ncols):
        iob_matrix[:, t_pre] = np.roll(iob_matrix[:, t_pre], t_pre)

    # Step 5: Fill the upper triangle with zeros - CS - is this necessary?
    iob_matrix_tri = iob_matrix * np.tri(nrows, ncols, 0)

    # Step 6: Sum across the curves to get the iob at every time step
    iob_sbr_t = np.sum(iob_matrix_tri, axis=1)

    # Step 7: Just get the last 8 hours
    iob_sbr_t = iob_sbr_t[95:-1]

    return iob_sbr_t


def get_steady_state_iob_from_sbr(sbr):
    """
    Get the steady state insulin on board for a given scheduled basal rate. This is
    the iob once the basal insulin stacking and metabolism clearing reach equilibrium.
    Parameters
    ----------
    sbr

    Returns
    -------

    """
    return sbr * STEADY_STATE_IOB_FACTOR_FDA


get_iob_from_sbr(1.0)
