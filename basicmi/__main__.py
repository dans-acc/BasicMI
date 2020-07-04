import mne

import pathlib

import numpy as np
np.random.seed(2435)

import EEGLearn.utils as eeg_utils

from basicmi import utils


def gen_subj_img(cap_coords, samples_x_features_mtx, n_grid_points=32, normalise=True, edgeless=False):

    # TODO: Should return a dictionary!!! gen_proj_imgs, gen_subj_img

    # Convert list of types to np.array (if not already).
    if not isinstance(cap_coords, np.ndarray):
        cap_coords = np.asarray(cap_coords)
    if not isinstance(samples_x_features_mtx, np.ndarray):
        samples_x_features_mtx = np.asarray(samples_x_features_mtx)

    # Delegate image generation to the tf_EEGLean library.
    return eeg_utils.gen_images(locs=cap_coords, features=samples_x_features_mtx,
                                n_gridpoints=n_grid_points, normalize=normalise,
                                edgeless=edgeless)


def gen_unpacked_folds(proj_ids, proj_epochs, proj_feats, as_np_arr=True):

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
    for sid in proj_ids:

        # Unpack the subjects ids and features.
        subj_feats = proj_feats[sid]
        ids += [sid for i in range(len(subj_feats))]
        samples += subj_feats

        # Ensure mapping validity.
        assert len(ids) == len(samples)

    # Generate pairs of index values, representing the training and test sets.
    folds = []
    for sid in proj_ids:

        # Generate test (selected ids) and training (not selected ids) sets based on the selected sid samples.
        selected_sid_samples = ids == sid
        training_set_indices = np.squeeze(np.nonzero(np.bitwise_not(selected_sid_samples)))
        test_set_indices = np.squeeze(np.nonzero(selected_sid_samples))

        # Shuffles only the index values within each respective array.
        np.random.shuffle(training_set_indices)
        np.random.shuffle(test_set_indices)

        # Add the pairs to the list of folds.
        folds.append((training_set_indices, test_set_indices))

    return np.asarray(folds) if as_np_arr else folds


def main():

    # Load the cap montage (2D coordinates).
    neuroscan_coords = utils.get_neuroscan_montage(azim=True, as_np_arr=True)
    if neuroscan_coords is None:
        return

    # Load the subjects into memory.
    proj_epochs = utils.get_proj_epochs(subj_ids=[1, 2],
                                        equalise_event_ids=['Left', 'Right', 'Bimanual'],
                                        inc_subj_info_id=True)

    if proj_epochs is None or not proj_epochs.keys():
        return

    # Generate the fold pairs according to leave-one-out validation.
    folds = gen_unpacked_folds(proj_ids=[1, 2],
                               proj_epochs=proj_epochs,
                               proj_feats={1: [1, 2, 3, 4, 5], 2: [6, 7, 8, 9, 10]},
                               as_np_arr=True)

    if folds is None:
        return

    print(folds)

if __name__ == '__main__':
    main()


    """
    subj_data, subj_labels = tools.get_epochs_data_and_labels(epochs=subj_epochs, data=False)
    freq_bands = [(4, 7), (8, 13), (13, 30)]
    samples_x_features_mtx = tools.gen_epochs_psd_features(epochs=subj_epochs,
                                                           t_min=0,
                                                           t_max=5,
                                                           freq_bands=freq_bands,
                                                           n_jobs=2,
                                                           include_classes=False,
                                                           as_np_arr=True)                                                           
    print('The samples_x_features_mtx is:')
    print(samples_x_features_mtx.shape)
    """

    """
    print('^' * 100)
    print(np.asarray(samples_x_features_mtx).shape)
    print('v' * 100)

    print('Sample len: %d' % len(samples_x_features_mtx))
    print('Feature len: %d' % len(samples_x_features_mtx[0]))
    print(np.asarray(samples_x_features_mtx))
    """
    """
    # Get the neuroscan montage locations and project them onto a 2D layout.
    neuroscan_coords = tools.get_neuroscan_montage(azim=True)
    neuroscan_coords = np.asarray(neuroscan_coords)

    # Generate images based on the feature matrix, coordinates, etc.
    subj_images = {}
    subj_images[1] = images = tools.gen_images(cap_locations=neuroscan_coords,
                                               samples_x_features_mtx=samples_x_features_mtx,
                                               n_grid_points=32,
                                               normalise=True,
                                               edgeless=False)
    """

    """
    tools.gen_folds(subj_ids=[1],
                    subj_epochs=proj_epochs,
                    epoch_feats=subj_images,
                    as_np_arr=True)
    """

    """
    images = tools.gen_images(cap_locations=neuroscan_coords,
                              samples_x_features_mtx=features,
                              n_grid_points=32,
                              normalise=True,
                              edgeless=False)
    """

    """

    Delta band is < 4Hz
    Theta band is 4-8Hz;
    Alpha band is 8-12Hz;
    Beta band is 13-30Hz;
    Gamma band is > 30Hz

    :param epochs:
    :param freq_bands: follow the form (f_min, f_max) in order theta, alpha, beta, gamma
    :return:
    """

