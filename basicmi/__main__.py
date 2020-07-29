import EEGLearn.utils as eeg_utils
import EEGLearn.train as eeg_train

import mne
import tensorflow as tf
import numpy as np

from basicmi import tools
from basicmi import feats


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


def gen_subj_win_img(subj_feats, cap_coords, n_grid_points=32, normalise=True, edgeless=False):
    win_imgs = []
    for i in range(len(subj_feats)):
        # TODO: features should not be extracted as such. There is a clear dependency when unfolding.
        win_imgs.append(gen_subj_imgs(subj_feats=subj_feats[i], cap_coords=cap_coords, n_grid_points=n_grid_points,
                                      normalise=normalise, edgeless=edgeless))
    return np.array(win_imgs)


def gen_proj_win_imgs(proj_ids, proj_feats, cap_coords, n_grid_points=32, normalise=True, edgeless=False):
    proj_imgs = {}
    for sid in np.unique(proj_ids):
        subj_imgs = gen_subj_win_img(subj_feats=proj_feats[sid], cap_coords=cap_coords, n_grid_points=n_grid_points,
                                     normalise=normalise, edgeless=edgeless)
        if subj_imgs is None:
            raise RuntimeError('Unable to generate images for subject: %d' % sid)
        print('Adding img win: %s' % subj_imgs)
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
        # TODO: might have to access labels with subj_feats[0]. First dimen represents the windows. CHECK THIS!
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

    # Init lib pseudo-random generators.
    np.random.seed(2435)
    tf.set_random_seed(2435)

    # Load the electrode cap, implicitly projecting 3D positions to 2D.
    neuroscan_coords = tools.get_neuroscan_montage(azim=True, as_np_arr=True)
    if neuroscan_coords is None:
        return

    # The list of subjects used for training.
    subjects = [1, 2, 3, 4, 5, 6]

    # Load all of the epochs into memory (from .fif files).
    proj_epochs = tools.get_proj_epochs(subj_ids=subjects, equalise_event_ids=['Left', 'Right', 'Bimanual'],
                                        inc_subj_info_id=True)
    if proj_epochs is None or not proj_epochs.keys():
        return

    # Generate the feature vectors for each of the subjects epochs.
    freq_bands = [(4, 8), (8, 13), (13, 30)]
    proj_feats = feats.extract_proj_psd_feats(proj_epochs=proj_epochs, t_min=0, t_max=5, freq_bands=freq_bands,
                                              n_jobs=3, inc_classes=False, as_np_arr=True)

    """
    TODO: move the window generating code to the utils file.
    time_windows = []
    for time in range(0, 10):
        t = time * 0.5
        time_windows.append((t, t + 0.5))
    print('Time windows: %s' % time_windows)

    # TODO: dont call this directly. Instead, ram this into a for loop.
    proj_feats = feats.extract_proj_win_psd_feats(proj_epochs=proj_epochs, windows=time_windows, freq_bands=freq_bands,
                                                  n_jobs=20, inc_classes=False, as_np_arr=True)
    """

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
