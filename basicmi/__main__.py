import mne

import pathlib

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

    print('+' * 100)
    print(features)
    print('+' * 100)
    print(len(features))
    print('+' * 100)
    print(len(features[0]))
    print('+' * 100)

    # Get the projects epochs.
    proj_epochs = tools.get_proj_epochs(subj_ids=[1],
                                        equalise_event_ids=['Left', 'Right', 'Bimanual'])

    # Get the subject data.
    subj_epochs = proj_epochs[1]

    print('Subject has %d epochs' % len(subj_epochs))

    subj_data, subj_labels = tools.get_epochs_data_and_labels(epochs=subj_epochs, data=False)

    print('±' * 200)
    print(subj_data)
    print(subj_labels)
    print('±' * 200)

    freq_bands = [(4, 5)]
    feature_mtx = tools.get_epochs_psd_features(subj_epochs, 0, 5, freq_bands=freq_bands, n_jobs=2)
    print('The Generated feature matrix is:')
    print(feature_mtx)

    # Get the neuroscan montage locations.
    neuroscan_coords = tools.get_neuroscan_montage(azim=True)
    #print(neuroscan_coords)

    # Testing image generation.
