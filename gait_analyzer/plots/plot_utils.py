from typing import Tuple, Any
from enum import Enum

import numpy as np
from scipy.interpolate import CubicSpline


class LegToPlot(Enum):
    LEFT = "L"
    RIGHT = "R"
    BOTH = "both"
    DOMINANT = "dominant"


class PlotType(Enum):
    Q = "q"
    QDOT = "qdot"
    QDDOT = "qddot"
    TAU = "tau"
    POWER = "power"
    ANGULAR_MOMENTUM = "h"


class DimentionsToPlot(Enum):
    BIDIMENTIONAL = "2D"
    TRIDIMENTIONAL = "3D"


def get_unit_conversion_factor(plot_type: PlotType, subject_mass: float | None) -> Tuple[float, str]:
    """
    This function returns the unit conversion factor for the plot type.
    .
    Parameters
    ----------
    plot_type: PlotType
        The type of plot to get the units for
    subject_mass: float
        The mass of the subject in kg
    """
    if plot_type == PlotType.Q:
        unit_conversion = 180 / np.pi
    elif plot_type == PlotType.QDOT:
        unit_conversion = 180 / np.pi
    elif plot_type == PlotType.QDDOT:
        unit_conversion = 180 / np.pi
    elif plot_type == PlotType.TAU:
        unit_conversion = 1/subject_mass
    elif plot_type == PlotType.POWER:
        unit_conversion = 1/subject_mass
    elif plot_type == PlotType.ANGULAR_MOMENTUM:
        unit_conversion = 1
    else:
        raise ValueError("plot_type must be a PlotType.")
    return unit_conversion


def get_unit_names(plot_type: PlotType) -> Tuple[float, str]:
    """
    This function returns the unit string for the plot type.
    .
    Parameters
    ----------
    plot_type: PlotType
        The type of plot to get the units for
    """
    if plot_type == PlotType.Q:
        unit_str = r"[$^\circ$]"
    elif plot_type == PlotType.QDOT:
        unit_str = r"[$^\circ/s$]"
    elif plot_type == PlotType.QDDOT:
        unit_str = r"[$^\circ/s^2$]"
    elif plot_type == PlotType.TAU:
        unit_str = r"[$Nm/kg$]"
    elif plot_type == PlotType.POWER:
        unit_str = r"[$W/kg$]"
    elif plot_type == PlotType.ANGULAR_MOMENTUM:
        unit_str = r"[$kg.m^2/s$]"
    else:
        raise ValueError("plot_type must be a PlotType.")
    return unit_str


def split_cycles(data: np.ndarray, event_idx: list[int], plot_type: PlotType, subject_mass: float) -> list[np.ndarray]:
    """
    This function splits the data into cycles at the event.
    .
    Parameters
    ----------
    data: np.ndarray (data_dim, frames_dim)
        The data to split into cycles
    event_idx: list[int]
        The index of the events
    plot_type: PlotType
        The type of plot to split the data for
    subject_mass: float
        The mass of the subject in kg
    .
    Returns
    -------
    cycles: list[np.ndarray] nb_cycles x (data_dim, frames_dim)
        The data split into cycles
    """
    # Checks
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("data must be a 2D numpy array.")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("data must not be empty.")
    if data.shape[1] < event_idx[-1]:
        raise RuntimeError(
            f"Watch out, you are trying to plot data shape {data.shape}, and the code expects shape (nb_data_dim, nb_frames)."
            f"Your frame dimension {data.shape[1]} is too short for the event indices {event_idx}."
        )

    unit_conversion = get_unit_conversion_factor(plot_type, subject_mass)
    # Split the data into cycles (skipping everything before the first event and after the last event)
    cycles = []
    for i_event in range(len(event_idx) - 1):
        current_cycle = data[:, event_idx[i_event] : event_idx[i_event + 1]]
        cycles += [current_cycle * unit_conversion]

    return cycles


def mean_cycles(
    data: list[np.ndarray], index_to_keep: list[int], nb_frames_interp: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function computes the mean over cycles.

    Parameters
    ----------
    data: list[np.ndarray] nb_cycles x (data_dim, frames_dim)
        The data to compute the mean of the cycles
    index_to_keep: list[int]
        The index of the data to perform the mean on
    nb_frames_interp: int
        The number of frames to interpolate the data on

    Returns
    -------
    mean_data: np.ndarray (data_dim, nb_frames_interp)
        The mean across cycles
    std_data: np.ndarray (data_dim, nb_frames_interp)
        The std across cycles
    """
    # Checks
    if not isinstance(data, list):
        raise ValueError("data must be a list.")
    if not isinstance(nb_frames_interp, int):
        raise ValueError("nb_frames_interp must be an integer.")
    if len(data) == 0:
        raise ValueError("data must not be empty.")

    data_dim = len(index_to_keep)
    interpolated_data_array = np.zeros((len(data), data_dim, nb_frames_interp))
    fig_data_dim = data[0].shape[0]
    for i_cycle, cycle in enumerate(data):
        if fig_data_dim != cycle.shape[0]:
            raise ValueError(f"Data dimension is inconsistant across cycles.")
        frames_dim = cycle.shape[1]
        # TODO: @ThomasAout -> How do you usually deal with the cycle length being variable ?
        x_to_interpolate_on = np.linspace(0, 1, num=nb_frames_interp)
        for i_dim, dim in enumerate(index_to_keep):
            y_data_old = cycle[dim, :]
            x_data = np.linspace(0, 1, num=frames_dim)
            y_data = y_data_old[~np.isnan(y_data_old)]
            x_data = x_data[~np.isnan(y_data_old)]
            interpolation_object = CubicSpline(x_data, y_data)
            interpolated_data_array[i_cycle, i_dim, :] = interpolation_object(x_to_interpolate_on)

    mean_data = np.nanmean(interpolated_data_array, axis=0)
    std_data = np.nanstd(interpolated_data_array, axis=0)

    return mean_data, std_data
