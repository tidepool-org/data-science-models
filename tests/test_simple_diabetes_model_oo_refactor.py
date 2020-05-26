"""
Tests to confirm that Cameron's refactor of Ed's diabetes model code gives the same output
"""

import numpy as np

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_models.models.simple_metabolism_OLD import (
    simple_metabolism_model,
    get_iob_from_sbr,
    get_steady_state_iob_from_sbr,
)
from tidepool_data_science_models.utils import EPSILON_TEST


def test_simple_metabolism_model_class():
    """
    Does the simple metabolism model class give that same output as Ed's original simple metabolism model function?
    """
    carb_amount = 0
    insulin_amount = 1.0
    cir = 10
    isf = 100

    (
        net_change_in_bg_func,
        t_5min_func,
        carb_amount_func,
        insulin_amount_func,
        iob_5min_func,
    ) = simple_metabolism_model(
        carb_amount=carb_amount, insulin_amount=insulin_amount, cir=cir, isf=isf
    )

    # Instantiate the class with the same values
    smm = SimpleMetabolismModel(
        insulin_sensitivity_factor=isf,
        carb_insulin_ratio=cir,
        insulin_model_name="palerm",
        carb_model_name="cescon",
    )

    (
        net_change_in_bg_smm,
        t_5min_smm,
        carb_amount_smm,
        insulin_amount_smm,
        iob_5min_smm,
    ) = smm.run(num_hours=8, carb_amount=carb_amount, insulin_amount=insulin_amount)

    # Make sure there is "stuff" in there
    assert len(net_change_in_bg_func) != 0
    assert sum(net_change_in_bg_func) != 0

    # Make sure the refactor gives the same result
    assert np.array_equal(net_change_in_bg_func, net_change_in_bg_smm)
    assert np.array_equal(t_5min_func, t_5min_smm)
    assert np.array_equal(carb_amount_func, carb_amount_smm)
    assert np.array_equal(insulin_amount_func, insulin_amount_smm)
    assert np.array_equal(iob_5min_func, iob_5min_smm)


def test_simple_metabolism_model_class_iob_sbr():

    cir = 10
    isf = 100

    smm = SimpleMetabolismModel(
        insulin_sensitivity_factor=isf,
        carb_insulin_ratio=cir,
        insulin_model_name="palerm",
        carb_model_name="cescon",
    )

    for sbr in [0, 0.1, 1.0, 10.0]:
        iob_t_class = smm.get_iob_from_sbr(sbr)
        iob_t_func = get_iob_from_sbr(sbr)
        assert np.array_equal(iob_t_class, iob_t_func)


def test_simple_metabolism_model_class_iob_steady_state():

    cir = 10
    isf = 100

    smm = SimpleMetabolismModel(
        insulin_sensitivity_factor=isf,
        carb_insulin_ratio=cir,
        insulin_model_name="palerm",
        carb_model_name="cescon",
    )

    for sbr in [0, 0.1, 1.0, 10.0]:
        sbr_steady_state_class = smm.get_steady_state_iob_from_sbr(
            sbr, use_fda_submission_constant=False
        )
        sbr_steady_state_func = get_steady_state_iob_from_sbr(sbr)

        assert abs(sbr_steady_state_class - sbr_steady_state_func) < EPSILON_TEST

        sbr_steady_state_class = smm.get_steady_state_iob_from_sbr(
            sbr, use_fda_submission_constant=True
        )

        assert abs(sbr_steady_state_class - sbr_steady_state_func) < EPSILON_TEST
