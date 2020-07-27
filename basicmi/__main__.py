import EEGLearn.utils as eeg_utils
import EEGLearn.train as eeg_train


import mne
import tensorflow as tf
import numpy as np
np.random.seed(2435)


from basicmi import tools


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


def gen_subj_imgs(subj_feats, cap_coords, n_grid_points=32, normalise=True, edgeless=False):

    # Convert the necessary parameters to np arrays (if not already).
    if not isinstance(subj_feats, np.ndarray):
        subj_feats = np.asarray(subj_feats)
    if not isinstance(cap_coords, np.ndarray):
        cap_coords = np.asarray(cap_coords)

    # Delegate the task of generating subject images to the tf_EEGLearn lib.
    return eeg_utils.gen_images(locs=cap_coords, features=subj_feats,
                                n_gridpoints=n_grid_points, normalize=normalise,
                                edgeless=edgeless)


def gen_proj_imgs(proj_ids, proj_feats, cap_coords, n_grid_points=32, normalise=True, edgeless=False):

    # Validate the parameters.
    if proj_ids is None or proj_feats is None or cap_coords is None:
        raise ValueError('Parameter(s) are None.')
    elif not proj_ids:
        return []

    # Generate images for each of the subjects.
    proj_imgs = {}
    for sid in np.unique(proj_ids):
        subj_imgs = gen_subj_imgs(subj_feats=proj_feats[sid], cap_coords=cap_coords, n_grid_points=n_grid_points,
                                  normalise=normalise, edgeless=edgeless)
        if subj_imgs is None:
            raise RuntimeError('Unable to generate images for subject: %d' % sid)
        proj_imgs[sid] = subj_imgs

    return proj_imgs


def unpacked_folds(proj_ids, proj_epochs, proj_feats, as_np_arr=True):

    # Validate the parameters.
    if proj_ids is None or proj_epochs is None or proj_feats is None:
        raise ValueError('Parameter(s) are None.')
    elif not proj_ids:
        return []

    # Get a sorted list of unique ids.
    proj_ids = np.sort(np.unique(proj_ids))

    # Unpack the subject ids and samples.
    ids = []
    samples = []
    labels = []
    for sid in proj_ids:

        # Unpack the subjects ids and features.
        subj_feats = proj_feats[sid]
        ids += [sid for i in range(len(subj_feats))]
        for subj_feat in subj_feats:
            samples.append(subj_feat)

        # Unpack the labels for the ids and samples.
        subj_epochs = proj_epochs[sid]
        subj_epochs_labels = tools.get_epochs_labels(epochs=subj_epochs)
        for subj_epoch_label in subj_epochs_labels:
            labels.append(subj_epoch_label)

        assert len(ids) == len(samples) == len(labels)

    # Generate pairs of index values, representing the training and test sets.
    folds = []
    for sid in proj_ids:

        # Generate test (selected ids) and training (not selected ids) sets based on the selected sid samples.
        selected_sid_samples = ids == sid
        training_set_indices = np.squeeze(np.nonzero(np.bitwise_not(selected_sid_samples)))
        test_set_indices = np.squeeze(np.nonzero(selected_sid_samples))

        print('+' * 100)
        print(len(training_set_indices))
        print(len(test_set_indices))

        # Shuffles only the index values within each respective array.
        np.random.shuffle(training_set_indices)
        np.random.shuffle(test_set_indices)

        # Add the pairs to the list of folds.
        folds.append((np.array(training_set_indices), np.array(test_set_indices)))

    return ids, samples, labels, folds


def main():

    # Load the electrode cap, implicitly projecting 3D positions to 2D.
    neuroscan_coords = tools.get_neuroscan_montage(azim=True, as_np_arr=True)
    if neuroscan_coords is None:
        return

    # The list of subjects used for training.
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # Load all of the epochs into memory (from .fif files).
    proj_epochs = tools.get_proj_epochs(subj_ids=subjects, equalise_event_ids=['Left', 'Right', 'Bimanual'],
                                        inc_subj_info_id=True)
    if proj_epochs is None or not proj_epochs.keys():
        return

    # Generate the feature vectors for each of the subjects epochs.
    freq_bands = [(4, 7), (8, 13), (13, 30)]
    proj_feats = extract_proj_psd_feats(proj_epochs=proj_epochs, t_min=0, t_max=1, freq_bands=freq_bands,
                                        n_jobs=3, inc_classes=False, as_np_arr=True)
    if proj_feats is None:
        return

    # Generate images from the feature maps.
    proj_imgs = gen_proj_imgs(proj_ids=subjects,
                              proj_feats=proj_feats,
                              cap_coords=neuroscan_coords,
                              n_grid_points=32,
                              normalise=True,
                              edgeless=False)
    if proj_imgs is None:
        return

    # Generate the fold pairs according to leave-one-out validation.
    ids, samples, labels, folds = unpacked_folds(proj_ids=subjects,
                                                 proj_epochs=proj_epochs,
                                                 proj_feats=proj_imgs,
                                                 as_np_arr=True)
    if folds is None:
        return

    print('+-' * 100)

    for f in folds:
        print('%d %d' % (len(f[0]), len(f[1])))

    print('+-' * 100)

    # Convert some types to np array.
    samples = np.array(samples)

    # Solves the batching issue.
    samples = np.expand_dims(samples, axis=0)

    # Labels must apparently start from 0.
    labels = [label-1 for label in labels]
    print(labels)

    labels = np.array(labels)
    folds = np.array(folds)

    print(samples.shape)

    print(type(samples))
    print(type(folds))

    # Train a basic, single-frame CNN.
    cnn_accuracy = []
    for i in range(len(folds)):
        cnn_accuracy.append(eeg_train.train(images=samples,
                                            labels=labels,
                                            fold=folds[i],
                                            model_type='cnn',
                                            batch_size=32,
                                            num_epochs=60))
        tf.reset_default_graph()


if __name__ == '__main__':
    main()
