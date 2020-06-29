import pathlib

import mne
import scipy.io as sio

# Commonly used paths relative to the tools.py file.
PROJ_DIR_PATH = pathlib.Path(pathlib.Path(__file__).parent.parent)
PROJ_DATA_DIR_PATH = pathlib.Path(PROJ_DIR_PATH.joinpath('data'))
PROJ_MI_DATA_DIR_PATH = pathlib.Path(PROJ_DATA_DIR_PATH.joinpath('mi'))


# Installed MNE dir path.
MNE_DIR_PATH = pathlib.Path(mne.__file__)

# Montages and layouts define 3D and 2D electrode positions, respectively.
MNE_LAYOUTS_DIR_PATH = pathlib.Path(MNE_DIR_PATH.joinpath('channels/data/layouts'))
MNE_MONTAGES_DIR_PATH = pathlib.Path(MNE_DIR_PATH.joinpath('channels/data/montages'))


def get_subj_epochs(subj_id, preload=True, equalise_events_ids=None):

    # Sanity checks.
    if subj_id is None:
        return None

    # Get the subject path.
    subj_path = pathlib.Path(PROJ_MI_DATA_DIR_PATH.joinpath('S%d//epochs_epo.fif' % subj_id))
    if not subj_path.exists() or not subj_path.is_file() or subj_path.is_dir():
        return None

    # Get the subjects epochs.
    epochs = mne.read_epochs(str(subj_path.absolute()), preload=preload)
    if epochs is None:
        return None

    # Whether the classes should be equalised.
    if equalise_events_ids is not None:
        epochs.equalize_event_counts(event_ids=equalise_events_ids)

    return epochs


def get_proj_epochs(subj_ids, preload=True, equalise_events_ids=None):

    # Sanity checks.
    if subj_ids is None:
        return None
    elif not subj_ids:
        return {}

    # Get epochs for each of the subjects.
    subj_epochs = {}
    for subj_id in subj_ids:
        subj_epoch = get_subj_epochs(subj_id=subj_id, preload=preload, equalise_events_ids=equalise_events_ids)
        if subj_epoch is None:
            print('Its none')
            continue
        subj_epochs[subj_id] = subj_epoch

    return subj_epochs


def concat_epochs(epochs, add_offset=False, equalise_event_ids=None):

    # Copy all epochs (to avoid side effects); concat into one epoch.
    copied_epochs = [epoch.copy() for epoch in epochs]
    concat = mne.concatenate_epochs(epochs_list=copied_epochs, add_offset=add_offset)

    # The class IDs that are to be equalised.
    if equalise_event_ids is not None:
        concat.equalize_event_counts(event_ids=equalise_event_ids)

    return concat


def get_montage(mne_montage='biosemi64'):
    return mne.channels.make_standard_montage(kind=mne_montage)


def set_epoch_montage(epoch, montage, copy=True):

    # Check epoch and montage are present.
    if epoch is None or montage is None:
        raise ValueError('Epoch or montage parameters are none.')

    # Potentially copy to avoid side-effects.
    if copy:
        epoch = epoch.copy()
        montage = montage.copy()

    # Apply the montage and return [the copies].]
    epoch.set_montage(montage=montage)
    return epoch, montage





def extract_data(epochs):

    # Get the epoch data as Trials x Channels x Samples; get the trial epoch class labels.
    if epochs is not None:
        return epochs.get_data(), epochs.events[:, 2]
    return None, None








