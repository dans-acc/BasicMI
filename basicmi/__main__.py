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

    # Get the projects epochs.
    proj_epochs = tools.get_proj_epochs(subj_ids=[1],
                                        equalise_event_ids=['Left', 'Right', 'Bimanual'])

    # Get the subject data.
    subj_epochs = proj_epochs[1]
    subj_data, subj_labels = tools.get_epoch_data_or_labels(epochs=subj_epochs, data=False)

    print('±' * 200)
    print(subj_data)
    print(subj_labels)
    print('±' * 200)

    freq_bands = [(4, 8)]
    feature_mtx = tools.get_epochs_psd_features(subj_epochs, 0, 5, freq_bands=freq_bands)

    # Get the neuroscan montage locations.
    neuroscan_coords = tools.get_neuroscan_montage(azim=True)
    print(neuroscan_coords)

"""

EEG_LEARN_DIR_PATH = pathlib.Path(eeg_learn.__file__)

for path in EEG_LEARN_DIR_PATH.parent.glob('*'):
    print(path)
print('±' * 100)

subj_epochs = tools.get_proj_epochs(subj_ids=[1],
                                    equalise_events_ids=['Left', 'Right', 'Bimanual'])

print(type(subj_epochs[1]))

subj_1_epochs = subj_epochs[1]

epoch, montage = tools.set_epoch_mne_montage(subj_1_epochs, mne_montage='standard_alphabetic')
if epoch is None or montage is None:
    print('One of them is none?')

print(epoch.info)
"""

"""
montage_fig = montage.plot(kind='3d')
montage_fig.gca().view_init(azim=70, elev=15)
montage.plot(kind='topomap', show_names=False)
"""