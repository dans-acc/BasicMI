import logging

from typing import List, Dict, Callable, Tuple

import mne
import numpy as np


from basicmi import utils


_logger = utils.create_logger(__name__, level=logging.DEBUG)


def get_subject_psd_feats(subject_epochs, t_min, t_max, freq_bands, n_jobs=3, append_classes=False, as_np_arr=True):

    # Feature matrix represents samples (epochs) * features (theta, alpha and beta bands).
    samples_x_feats_mtx = []
    for epoch_index in range(len(subject_epochs)):
        samples_x_feats_mtx.append([])

    # Generate FFT PSD features for each of the epochs within subject_epochs.
    for f_min, f_max in freq_bands:

        # Returns a matrix in the shape of n_epochs, n_channels, n_freqs.
        psds, freqs = mne.time_frequency.psd_multitaper(inst=subject_epochs, tmin=t_min, tmax=t_max, fmin=f_min,
                                                        fmax=f_max, proj=True, n_jobs=n_jobs)
        if psds is None or freqs is None:
            raise ValueError('Unable to generate power spectral density features.')

        # Loop through each epoch index (from within subject_epochs), then each channel,
        # adding the mean FFT PSD to the feature matrix.
        for epoch_index in range(len(psds)):
            for channel in psds[epoch_index]:
                mean_channel_psds = channel.mean()
                samples_x_feats_mtx[epoch_index].append(mean_channel_psds)

    # Whether classes should be appended to the end of the matrix.
    if not append_classes:
        _logger.debug('Returning samples_x_feats matrix. No classes are appended to the end.')
        return np.array(samples_x_feats_mtx) if as_np_arr else samples_x_feats_mtx

    # Return the matrix containing the samples x features (with classes appended to the end of the features.)
    return np.array(samples_x_feats_mtx) if as_np_arr else samples_x_feats_mtx


def get_psds_feats(epochs, t_min, t_max, freq_bands, n_jobs=3, append_classes=False, as_np_arr=True):
    subject_features = {}
    for subject_id, subject_epochs in epochs.items():
        _logger.info('Generating PSDS features for subject: %d', subject_id)
        subject_features[subject_id] = get_subject_psd_feats(subject_epochs=subject_epochs, t_min=t_min, t_max=t_max,
                                                             freq_bands=freq_bands, n_jobs=n_jobs,
                                                             append_classes=append_classes, as_np_arr=as_np_arr)
    return subject_features


def get_psd_features(subject_epochs: mne.Epochs, windows: List[Tuple[float, float]], bands: List[Tuple[float, float]],
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

            # Returns a matrix in the shape of n_epochs, n_channels, n_freqs.
            _logger.debug('Generating PSD features for band: %s - %s', str(f_min), str(f_max))
            psds, freqs = mne.time_frequency.psd_multitaper(inst=subject_epochs, tmin=t_min, tmax=t_max,
                                                            fmin=f_min, fmax=f_max, proj=True, n_jobs=n_jobs)

            # Loop through each epoch index, then each channel, adding the mean FFT PSD to the feature matrix.
            for epoch_idx in range(len(psds)):
                for channel in psds[epoch_idx]:
                    mean_channel_psd = channel.mean()
                    mtx[window_idx][epoch_idx].append(mean_channel_psd)

    # Finally, convert the list matrix into a numpy matrix and return.
    np_mtx = np.asarray(mtx)
    _logger.debug('Generated feature matrix: %s', str(np_mtx.shape))
    return np_mtx


def get_features(epochs: Dict[int, mne.Epochs], target: Callable = None, **kwargs) -> Dict[int, np.ndarray]:
    features = {}
    for subject_id, subject_epochs in epochs.items():
        subject_features = target(**kwargs)
    return features
