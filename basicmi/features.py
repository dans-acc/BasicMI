import logging


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

    # Get the labels associated with each of the epochs.
    epochs_labels = utils.get_epochs_labels(epochs=subject_epochs)
    if epochs_labels is None:
        raise ValueError('Unable to get the associated labels for epoch: %s', str(subject_epochs))

    # Epochs and labels must be the same length.
    assert len(epochs_labels) == len(samples_x_feats_mtx)

    # Include the classes into the feature matrix.
    for i in range(len(samples_x_feats_mtx)):
        features = samples_x_feats_mtx[i]
        features.append(epochs_labels[i])

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
