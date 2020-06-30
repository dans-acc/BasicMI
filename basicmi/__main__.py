import mne

import pathlib

from basicmi import tools


if __name__ == '__main__':

    # Get the projects epochs.
    proj_epochs = tools.get_proj_epochs(subj_ids=[1, 2, 3],
                                        equalise_event_ids=['Left', 'Right', 'Bimanual'])

    # Get the subj epochs; set the montage.
    subj_epochs = proj_epochs[1]
    tools.set_epochs_mne_montage(epochs=subj_epochs, kind='mgh70')

    print('±' * 100)
    print(subj_epochs.info)

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