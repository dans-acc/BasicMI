import logging


import numpy as np


from basicmi import tools


# Logger is utilised to debug the feature extraction code.
_logger = tools.create_logger(__name__, level=logging.DEBUG)


def extract_subj_psd_feats(epochs, t_min, t_max, freq_bands, n_jobs=3, inc_classes=False, as_np_arr=True):

    # Validate the parameters.
    if epochs is None or freq_bands is None:
        return None
    elif not freq_bands:
        return np.asarray([]) if as_np_arr else []

    # The feature matrix represents samples (epochs) * features (i.e. theta, alpha and beta bands).
    samples_x_features_mtx = []
    for epoch_index in range(len(epochs)):
        samples_x_features_mtx.append([])

    # Generate FFT PSD features for each of the epochs.
    for f_min, f_max in freq_bands:

        # Returns a matrix in the shape of (n_epochs, n_channels, n_freqs)
        psds, freqs = mne.time_frequency.psd_multitaper(inst=epochs, tmin=t_min, tmax=t_max, fmin=f_min, fmax=f_max,
                                                        proj=True, n_jobs=n_jobs)

        print('Freqs: %d' % len(freqs))

        if psds is None or freqs is None:
            return None

        # Loop through each epoch index; then each channel; adding the mean FFT PSD to the feature matrix.
        for epoch_index in range(len(psds)):
            for channel in psds[epoch_index]:
                mean_channel_psds = channel.mean()
                samples_x_features_mtx[epoch_index].append(mean_channel_psds)

    # Present because further processing includes classes within the feature matrix.
    if not inc_classes:
        return np.asarray(samples_x_features_mtx) if as_np_arr else samples_x_features_mtx

    # Get the labels associated with each of the epochs.
    epochs_labels = tools.get_epochs_labels(epochs=epochs)
    if epochs_labels is None:
        return None

    # Epochs and labels must be the same length.
    assert len(epochs_labels) == len(samples_x_features_mtx)

    # Include the classes into the feature matrix.
    for i in range(len(samples_x_features_mtx)):
        features = samples_x_features_mtx[i]
        features.append(epochs_labels[i])

    # Return the feature matrix without the included classes.
    return np.asarray(samples_x_features_mtx) if as_np_arr else samples_x_features_mtx


def extract_proj_psd_feats(proj_epochs, t_min, t_max, freq_bands, n_jobs=3, inc_classes=False, as_np_arr=True):

    # Loop through each of the sid, epoch pairs, generating a feature matrix for each.
    proj_feats = {}
    for sid, subj_epochs in proj_epochs.items():
        proj_feats[sid] = extract_subj_psd_feats(epochs=subj_epochs, t_min=t_min, t_max=t_max, freq_bands=freq_bands,
                                                 n_jobs=n_jobs, inc_classes=inc_classes, as_np_arr=as_np_arr)

    return proj_feats
