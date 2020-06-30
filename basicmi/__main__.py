import mne

from basicmi import tools


subj_epochs = tools.get_proj_epochs(subj_ids=[1],
                                    equalise_events_ids=['Left', 'Right', 'Bimanual'])

print(type(subj_epochs[1]))

subj_1_epochs = subj_epochs[1]

epoch, montage = tools.set_epoch_mne_montage(subj_1_epochs, mne_montage='standard_alphabetic')
if epoch is None or montage is None:
    print('One of them is none?')

print(epoch.info)

"""
montage_fig = montage.plot(kind='3d')
montage_fig.gca().view_init(azim=70, elev=15)
montage.plot(kind='topomap', show_names=False)
"""