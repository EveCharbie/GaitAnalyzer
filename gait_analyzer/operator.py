import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


class Operator:

    @staticmethod
    def moving_average(x: np.array, window_size: int):
        """
        Compute the moving average of a signal
        .
        Parameters
        ----------
        x: np.array
            The signal to be averaged (for now, this can only be a vector)
        window_size: int
            The size of the window to compute the average on
        .
        Returns
        -------
        x_averaged: np.array
            The signal processed using the moving average
        """
        # Checks
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number")
        if len(x.shape) != 1:
            if len(x.shape) == 2 and x.shape[1] == 1:
                x = x.flatten()
            else:
                raise ValueError("x must be a vector")
        if x.shape[0] / 2 < window_size:
            raise ValueError("window_size must be smaller than half of the length of the signal")

        # Compute the moving average
        x_averaged = np.zeros_like(x)
        for i in range(len(x)):
            if i < window_size // 2:
                x_averaged[i] = np.mean(x[: i + window_size // 2 + 1])
            elif i >= len(x) - window_size // 2:
                x_averaged[i] = np.mean(x[i - window_size // 2 :])
            else:
                x_averaged[i] = np.mean(x[i - window_size // 2 : i + window_size // 2 + 1])
        return x_averaged

    @staticmethod
    def apply_filtfilt(data: np.ndarray, order: int, sampling_rate: int, cutoff_freq: int):
        """
        TODO: @ophlariviere -> This was taken from biomechanics tools, could you provide a ref for it ?
        .
        Apply a low-pass Butterworth filter to the data using scipy.filtfilt
        .
        Parameters
        ----------
        data: np.ndarray
            The data to be filtered (for now, this can only be a nb_data x nb_frames array)
        order: int
            The order of the Butterworth filter
        sampling_rate: int
            The sampling rate of the data in Hz
        cutoff_freq: int
            The cutoff frequency of the filter in Hz
        .
        Returns
        -------
        filtered_data: np.ndarray
            The filtered data
        """
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        filtered_data = np.zeros_like(data)
        filtered_data[:, :] = np.nan
        for i_data in range(data.shape[0]):
            non_nan_idx = ~np.isnan(data[i_data, :])
            filtered_data[i_data, non_nan_idx] = filtfilt(b, a, data[i_data, non_nan_idx], axis=0)
        return filtered_data

    @staticmethod
    def apply_savgol(data: np.ndarray, window_length: int, polyorder: int):
        """
        TODO: @ophlariviere -> This was taken from biomechanics tools, could you provide a ref for it ?
        .
        Apply a low-pass Savitzky-Golay filter to the data using scipy.savgol_filter
        .
        Parameters
        ----------
        data: np.ndarray
            The data to be filtered (for now, this can only be a nb_data x nb_frames array)
        window_length: int
            The length of the filter window
        polyorder: int
            The order of the polynomial to fit
        .
        Returns
        -------
        filtered_data: np.ndarray
            The filtered data
        """
        filtered_data = np.zeros_like(data)
        filtered_data[:, :] = np.nan
        for i_data in range(data.shape[0]):
            non_nan_idx = ~np.isnan(data[i_data, :])
            filtered_data[i_data, non_nan_idx] = savgol_filter(
                data[i_data, non_nan_idx], window_length=window_length, polyorder=polyorder, axis=0
            )
        return filtered_data

    @staticmethod
    def from_marker_frame_to_analog_frame(
        analogs_time_vector: np.ndarray, markers_time_vector: np.ndarray, marker_idx: int | list[int]
    ) -> int | list[int] | np.ndarray[int]:
        """
        This function converts a marker frame index into an analog frame index since the analogs are sampled at a higher frequency than the markers.
        .
        Parameters
        ----------
        analogs_time_vector: np.ndarray
            The time vector of the analogs
        markers_time_vector: np.ndarray
            The time vector of the markers
        marker_idx: int | list[int] | np.ndarray[int]
            The analog frame index to convert
        .
        Returns
        -------
        analog_idx: int | list[int] | np.ndarray[int]
            The analog frame index
        """
        analog_to_marker_ratio = int(round(analogs_time_vector.shape[0] / markers_time_vector.shape[0]))
        all_idx = list(range(0, len(analogs_time_vector), analog_to_marker_ratio))
        if isinstance(marker_idx, int):
            analog_idx = all_idx[marker_idx]
        elif isinstance(marker_idx, list):
            analog_idx = [all_idx[idx] for idx in marker_idx]
        elif isinstance(marker_idx, np.ndarray):
            if len(marker_idx.shape) != 1:
                raise ValueError("marker_idx must be a 1D numpy array.")
            analog_idx = np.zeros_like(marker_idx).astype(int)
            for i_idx in range(marker_idx.shape[0]):
                analog_idx[i_idx] = all_idx[marker_idx[i_idx]]
        else:
            raise ValueError("marker_idx must be an int or a list of int or a np.ndarray of int.")
        return analog_idx

    @staticmethod
    def from_analog_frame_to_marker_frame(
        analogs_time_vector: np.ndarray, markers_time_vector: np.ndarray, analog_idx: int | list[int] | np.ndarray[int]
    ) -> int | list[int] | np.ndarray[int]:
        """
        This function converts an analog frame index into a marker frame index since the analogs are sampled at a higher frequency than the markers.
        .
        Parameters
        ----------
        analogs_time_vector: np.ndarray
            The time vector of the analogs
        markers_time_vector: np.ndarray
            The time vector of the markers
        analog_idx: int | list[int] | np.ndarray[int]
            The marker frame index to convert
        .
        Returns
        -------
        marker_idx: int | list[int] | np.ndarray[int]
            The marker frame index
        """
        analog_to_marker_ratio = int(round(analogs_time_vector.shape[0] / markers_time_vector.shape[0]))
        if isinstance(analog_idx, int):
            marker_idx = int(round(analog_idx / analog_to_marker_ratio))
        elif isinstance(analog_idx, list):
            marker_idx = [int(round(idx / analog_to_marker_ratio)) for idx in analog_idx]
        elif isinstance(analog_idx, np.ndarray):
            if len(analog_idx.shape) != 1:
                raise ValueError("analog_idx must be a 1D numpy array.")
            marker_idx = np.zeros_like(analog_idx).astype(int)
            for i_idx in range(analog_idx.shape[0]):
                marker_idx[i_idx] = int(np.round(analog_idx[i_idx] / analog_to_marker_ratio))
        else:
            raise ValueError("analog_idx must be an int or a list of int or a np.ndarray of int.")
        return marker_idx
