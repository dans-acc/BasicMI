import logging

from typing import List, Dict, Callable, Tuple

import mne
import numpy as np


from basicmi import utils


_logger = utils.create_logger(__name__, level=logging.DEBUG)


def get_epoch_psd_features(subject_epochs: mne.Epochs, windows: List[Tuple[float, float]], bands: List[Tuple[float, float]],
                           n_jobs: int = 10) -> np.ndarray:

    # Feature matrix represents windows * samples (epochs) * features (theta, alpha, and beta bands)
    mtx = []
    for window_idx in range(len(windows)):
        samples = []
        for epoch_idx in range(len(subject_epochs)):
            samples.append([])
        mtx.append(samples)

    # Loop through each window and generate features.
    for window_idx in range(len(windows)):

        # For each of the windows generate PSDS features for the defined bands.
        t_min, t_max = windows[window_idx]
        _logger.info('Generating PSD features for window: %s - %s.', str(t_min), str(t_max))

        for f_min, f_max in bands:
            _logger.debug('Generating PSD features for band: %s - %s', str(f_min), str(f_max))

            # Returns a matrix in the shape of n_epochs, n_channels, n_freqs.
            psds, freqs = mne.time_frequency.psd_multitaper(inst=subject_epochs, tmin=t_min, tmax=t_max,
                                                            fmin=f_min, fmax=f_max, proj=True, n_jobs=n_jobs)

            # Loop through each epoch index, then each channel, adding the mean FFT PSD to the feature matrix.
            for epoch_idx in range(len(psds)):
                for channel in psds[epoch_idx]:
                    mean_channel_psd = channel.mean()
                    mtx[window_idx][epoch_idx].append(mean_channel_psd)

    np_mtx = np.asarray(mtx)
    _logger.debug('Generated feature matrix: %s', str(np_mtx.shape))
    return np_mtx


def get_psd_features(epochs: Dict[int, mne.Epochs], windows: List[Tuple[float, float]], bands: List[Tuple[float, float]],
                     n_jobs: int = 10) -> Dict[int, np.ndarray]:
    psd_features = {}
    for subject_id, subject_epochs in epochs.items():
        psd_features[subject_id] = get_epoch_psd_features(subject_epochs=subject_epochs, windows=windows,
                                                          bands=bands, n_jobs=n_jobs)
    return psd_features
