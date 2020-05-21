import numpy as np

# Some constants
STEADY_STATE_IOB_FACTOR_FDA = 2.111517
MINUTES_PER_HOUR = 60

# Small value for non-exact comparisons
EPSILON_TEST = 1e-5

# a value close to zero to check unit insulin decays
INSULIN_DECAY_8HR_EPSILON = 1e-2


def get_timeseries(num_hours, five_min=False):
    """
    Get a numpy array filled with values at every 1 minute or 5 minutes for the duration
    num_hours.

    Parameters
    ----------
    num_hours: float
        Number of hours for the time series

    five_min: bool
        Whether to get 5 minute increments

    Returns
    -------
    np.array
        The time series
    """
    minutes_in_model = int(num_hours * 60)  # hr * 60 minutes/hr

    if five_min:
        t = np.arange(0, minutes_in_model, 5)  # every 5 minutes
    else:
        t = np.arange(0, minutes_in_model, 1)  # in minutes

    return t


def get_figure_filename(short_name, version, dataset_name="simulated", extension="png"):
    """
    Get filename according to Data Science governance agreed format:

    <short-name>_<date>_<dataset-name>_<version>.<extension>

    Parameters
    ----------
    short_name: str
        Short description of the figure

    version: str
        Which version of code generated it

    dataset_name: str
        The data used in the figure

    extension: str
        The file extension

    Returns
    -------
    str:
        The figure name
    """
    return "{short_name}_{date}_{version}_{dataset_name}.{extension}".format(
        **{
            "short_name": short_name,
            "date": datetime.datetime.now().strftime("%Y-%m-%d_%H:%M-%p"),
            "version": version,
            "dataset_name": dataset_name,
            "extension": extension,
        }
    )
