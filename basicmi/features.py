import logging

from typing import List, Dict, Callable, Tuple

import mne
import numpy as np


from basicmi import utils


_logger = utils.create_logger(__name__, level=logging.DEBUG)


def get_epoch_psd_features(subject_epochs: mne.Epochs, windows: List[Tuple[float, float]], bands: List[Tuple[float, float]],
                           n_jobs: int = 10) -> np.ndarray:

    # Feature matrix represents windows * samples (epochs) * features (theta, alpha, and beta bands)
    mtx = [[[] for epoch_idx in range(len(subject_epochs))] for window_inx in range(len(windows))]

    # For each window, generate PSD features.
    for window_idx in range(len(windows)):

        t_min, t_max = windows[window_idx]
        _logger.debug('Generating PSD features for window: %s - %s.', str(t_min), str(t_max))

        # For each of the windows generate PSDS features for the defined bands.
        for f_min, f_max in bands:

            _logger.debug('Generating PSD features for band: %s - %s', str(f_min), str(f_max))

            # Returns a matrix in the shape of n_epochs, n_channels, n_freqs.
            psds, freqs = mne.time_frequency.psd_multitaper(inst=subject_epochs, tmin=t_min, tmax=t_max,
                                                            fmin=f_min, fmax=f_max, proj=False, n_jobs=n_jobs)

            # Loop through each epoch index, then each channel, adding the mean FFT PSD to the feature matrix.
            for epoch_idx in range(len(psds)):
                for channel in psds[epoch_idx]:
                    mean_channel_psd = channel.mean()
                    mtx[window_idx][epoch_idx].append(mean_channel_psd)

    return np.asarray(mtx)


def get_psd_features(epochs: Dict[int, mne.Epochs], windows: List[Tuple[float, float]], bands: List[Tuple[float, float]],
                     n_jobs: int = 10) -> Dict[int, np.ndarray]:

    # A dictionary containing all of the features.
    subject_features = {}

    # For each subject generate PSD features.
    for subject_id, subject_epochs in epochs.items():
        subject_features[subject_id] = get_epoch_psd_features(subject_epochs=subject_epochs, windows=windows,
                                                              bands=bands, n_jobs=n_jobs)
        _logger.info('Features %s for subject %d generated.', str(subject_features[subject_id].shape), subject_id)

    return subject_features


def concat_features(windows: int, epoch_feats: Dict[int, np.ndarray]) -> np.ndarray:

    # Contains all of the generated features concatenated together based on the window.
    features = [[] for window in range(windows)]

    # A list containing all of the unique IDs.
    unique_subject_ids = np.sort(np.unique(list(epoch_feats.keys())))

    # Loop through each of the windows, concatenating them together.
    for window in range(windows):
        for unique_subject_id in unique_subject_ids:

            # Because we are operating on a sorted list, all values must be present.
            if epoch_feats[unique_subject_id] is None:
                raise ValueError('Features for subject %d is None.', unique_subject_id)

            # Add the list of features to the must appropriate window.
            subject_features = epoch_feats[unique_subject_id]
            features[window].extend(subject_features[window])

    return np.array(features)
