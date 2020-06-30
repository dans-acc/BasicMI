import pathlib
import logging

import mne
import EEGLearn as eegl
import scipy.io as sio

# tools.py file path - used as a reference point.
PROJ_TOOLS_FILE_PATH = pathlib.Path(__file__)


# Project related paths (relative to the tools file path).
PROJ_DIR_PATH = pathlib.Path(PROJ_TOOLS_FILE_PATH.parent)
PROJ_RES_DIR_PATH = pathlib.Path(PROJ_DIR_PATH.joinpath('resources'))
PROJ_MONTAGES_DIR_PATH = pathlib.Path(PROJ_RES_DIR_PATH.joinpath('montages'))
PROJ_SUBJECTS_DIR_PATH = pathlib.Path(PROJ_RES_DIR_PATH.joinpath('subjects'))


# MNE resource related_paths
MNE_DIR_PATH = pathlib.Path(mne.__file__)
MNE_LAYOUTS_DIR_PATH = pathlib.Path(MNE_DIR_PATH.joinpath('channels/data/layouts'))
MNE_MONTAGES_DIR_PATH = pathlib.Path(MNE_DIR_PATH.joinpath('channels/data/montages'))


def get_mat_items(mat_path, mat_keys=None):

    if mat_path is None or not mat_path.exists() or not mat_path.is_file() or mat_path.is_dir():
        return None

    # Read the mat file.
    mat_file_dict = sio.loadmat(file_name=str(mat_path))
    if mat_file_dict is None:
        return None
    elif mat_keys is None:
        return mat_file_dict

    # Obtain only specific mat items (defined by mat_keys).
    return {key: mat_file_dict.get(key) for key in mat_keys}


def get_subj_epochs(subj_id, preload=True, equalise_event_ids=None):

    if subj_id is None:
        return None

    # Get the subject path based on the ID.
    subj_path = pathlib.Path(PROJ_SUBJECTS_DIR_PATH.joinpath('S%s//epochs_epo.fif' % str(subj_id)))
    if not subj_path.exists() or not subj_path.is_file() or subj_path.is_dir():
        return None

    # Read the subjects epochs.
    subj_epochs = mne.read_epochs(str(subj_path.absolute()), preload=preload)
    if subj_epochs is None:
        return None

    # If not none, equalise the defined classes.
    if equalise_event_ids is not None:
        subj_epochs.equalize_event_counts(event_ids=equalise_event_ids)

    return subj_epochs


def get_proj_epochs(subj_ids, preload=True, equalise_event_ids=None):

    if subj_ids is None:
        return None
    elif not subj_ids:
        return {}

    # Read all of the resource subject epochs.
    proj_epochs = {}
    for subj_id in subj_ids:
        subj_epochs = get_subj_epochs(subj_id=subj_id, preload=preload, equalise_event_ids=equalise_event_ids)
        if subj_epochs is None:
            continue
        proj_epochs[subj_id] = subj_epochs

    return proj_epochs


def concat_epochs(epochs, add_offset=False, equalise_event_ids=None):

    # Copy all subjects (to avoid side effects); concat into one epoch.
    copied_epochs = [epoch.copy() for epoch in epochs]
    concat = mne.concatenate_epochs(epochs_list=copied_epochs, add_offset=add_offset)

    # The class IDs that are to be equalised.
    if equalise_event_ids is not None:
        concat.equalize_event_counts(event_ids=equalise_event_ids)

    return concat


def get_mne_montage(kind):

    if kind is None:
        return None

    # Get a montage from MNE library.
    return mne.channels.make_standard_montage(kind=kind)


def set_epochs_mne_montage(epochs, kind, new=False):

    # Check parameter validity.
    if epochs is None or kind is None:
        return None, None

    # Get the montage.
    montage = get_mne_montage(kind=kind)
    if montage is None:
        return None, None

    # Set the epochs montage (on a potentially new epochs).
    epochs = epochs.copy() if new else epochs
    epochs.set_montage(montage=montage)

    # Return the epochs instance and the loaded montage.
    return epochs, montage


def extract_data(epochs):

    # Get the epoch data as Trials x Channels x Samples; get the trial epoch class labels.
    if epochs is not None:
        return epochs.get_data(), epochs.events[:, 2]
    return None, None








