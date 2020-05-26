"""
Testing the functionality of the simple diabetes model.
"""
from data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from data_science_models.utils import (
    EPSILON_TEST,
    INSULIN_DECAY_8HR_EPSILON,
    STEADY_STATE_IOB_FACTOR_FDA,
)


def test_simple_metabolism_model():
    """
    Test basic functionality
    """

    isf = 100
    cir = 10
    carb_amount = 0.0

    # Instantiate the class with the same values
    smm = SimpleMetabolismModel(
        insulin_sensitivity_factor=isf,
        carb_insulin_ratio=cir,
        insulin_model_name="palerm",
        carb_model_name="cescon",
    )

    for insulin_amount in [1.0, 10.0]:

        (net_change_in_bg, t_5min, carb_amount, insulin_amount, iob_5min,) = smm.run(
            carb_amount=carb_amount, insulin_amount=insulin_amount,
        )

        assert iob_5min[-1] < (INSULIN_DECAY_8HR_EPSILON * insulin_amount)
        assert len(net_change_in_bg) == 8 * 60 / 5
        assert iob_5min[0] == insulin_amount


def test_insulin_onboard_from_scheduled_basal_rate():
    isf = 100
    cir = 10

    smm = SimpleMetabolismModel(
        insulin_sensitivity_factor=isf,
        carb_insulin_ratio=cir,
        insulin_model_name="palerm",
        carb_model_name="cescon",
    )

    # Check several basal rates
    for scheduled_basal_rate in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:

        iob_t = smm.get_iob_from_sbr(sbr_actual=scheduled_basal_rate)

        # Should be 8 hours iob every 5 minutes
        assert len(iob_t) == (8 * 60 / 5)

        # Starting iob is correct
        assert (
            iob_t[0] - (scheduled_basal_rate * STEADY_STATE_IOB_FACTOR_FDA)
            < EPSILON_TEST
        )

        # Insulin should be mostly gone after 8 hours
        assert iob_t[-1] < INSULIN_DECAY_8HR_EPSILON
