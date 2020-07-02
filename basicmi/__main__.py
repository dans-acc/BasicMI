import mne

import pathlib

import numpy as np

from basicmi import tools

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


if __name__ == '__main__':

    features_items = tools.get_mat_items(tools.PROJ_MONTAGES_DIR_PATH.joinpath('FeatureMat_timeWin.mat'), mat_keys=['features'])
    features = features_items['features']

    # Get the projects epochs.
    proj_epochs = tools.get_proj_epochs(subj_ids=[1],
                                        equalise_event_ids=['Left', 'Right', 'Bimanual'])

    # Get the subject data.
    subj_epochs = proj_epochs[1]
    subj_data, subj_labels = tools.get_epochs_data_and_labels(epochs=subj_epochs, data=False)

    # Generate features for the subject from theta, alpha and beta bands.
    freq_bands = [(4, 7), (8, 13), (13, 30)]
    samples_x_features_mtx = tools.get_epochs_psd_features(subj_epochs, 0, 5, freq_bands=freq_bands, n_jobs=2)
    print('^' * 100)
    print(np.asarray(samples_x_features_mtx).shape)
    print('v' * 100)

    print('Sample len: %d' % len(samples_x_features_mtx))
    print('Feature len: %d' % len(samples_x_features_mtx[0]))
    print(np.asarray(samples_x_features_mtx))

    # Get the neuroscan montage locations and project them onto a 2D layout.
    neuroscan_coords = tools.get_neuroscan_montage(azim=True)
    neuroscan_coords = np.asarray(neuroscan_coords)

    print('Number of electrodes: %d' % neuroscan_coords.shape[0])
    print(neuroscan_coords.shape)

    # Generate images based on the feature matrix, coordinates, etc.
    """
    images = tools.gen_images(cap_locations=neuroscan_coords,
                              samples_x_features_mtx=samples_x_features_mtx,
                              n_grid_points=32,
                              normalise=True,
                              edgeless=False)
    """

    images = tools.gen_images(cap_locations=neuroscan_coords,
                              samples_x_features_mtx=features,
                              n_grid_points=32,
                              normalise=True,
                              edgeless=False)

    print('Images generated?')
