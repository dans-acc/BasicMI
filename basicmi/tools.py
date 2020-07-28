import pathlib
import logging

import mne
import EEGLearn.utils as eeg_utils
import numpy as np
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


def create_logger(name, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG):

    # Console Handler formatter.
    fmt = logging.Formatter(fmt=format)

    # Console handler (for displaying the messages on screen.)
    ch = logging.StreamHandler()
    ch.setLevel(level=level)
    ch.setFormatter(fmt=fmt)

    # Init the console logger.
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    logger.addHandler(ch)

    return logger


# For debugging tool code.
_logger = create_logger(name=__name__, level=logging.DEBUG)


def get_mat_items(mat_path, mat_keys=None):

    """
    Loads and returns key-value pairs from a mat file.
    :param mat_path: Path: the file path of the mat file that is to be loaded.
    :param mat_keys: list: contains the keys that are to be returned from the mat file.
    :return: dict: contains key-value pairs.
    """

    # Check that the mat file path is valid.
    if mat_path is None or not mat_path.exists() or not mat_path.is_file() or mat_path.is_dir():
        _logger.error('Invalid mat path: %s', mat_path)
        return None

    # Read the items from within the mat file.
    mat_file_dict = sio.loadmat(file_name=str(mat_path))
    if mat_file_dict is None:
        _logger.error('Unable to load mat file: %s', mat_path)
        return None
    elif mat_keys is None:
        _logger.debug('Parameter mat_keys is None. Returning all mat key-value pairs.')
        return mat_file_dict

    # Return only the defined mat keys from the file.
    _logger.debug('Parameter mat_keys is %s. Returning all mat_keys key-value pairs.', mat_keys)
    return {key: mat_file_dict.get(key) for key in mat_keys}


def get_subj_epochs(subj_id, preload=True, equalise_event_ids=None, inc_subj_info_id=True):

    """
    Load and instantiate a given subjects Epochs instance.
    :param subj_id: int: the ID associated with the subject.
    :param preload: bool: whether the epochs should be loaded directly into memory or load them only when required.
    :param equalise_event_ids: list: contains the epoch classes that should be equalised with respect to time.
    :param inc_subj_info_id: bool: whether additional subject ID data is to be generated and appended to the loaded
        Epochs instance.
    :return: Epochs: an instance containing the subjects epoch data.
    """

    # Check that the subject id is valid.
    if subj_id is None:
        _logger.error('Parameter subj_id is None.')
        return None

    # Get the subject .fif file path based on the ID.
    subj_path = pathlib.Path(PROJ_SUBJECTS_DIR_PATH.joinpath('S%s//epochs_epo.fif' % str(subj_id)))
    if not subj_path.exists() or not subj_path.is_file() or subj_path.is_dir():
        _logger.error('File path %s refers to an invalid subject file.', subj_path.absolute())
        return None

    # Read the subjects epochs data.
    subj_epochs = mne.read_epochs(str(subj_path.absolute()), preload=preload)
    if subj_epochs is None:
        _logger.error('Unable to read epoch data from file %s.', subj_path.absolute())
        return None

    # If not none, equalise the defined classes.
    if equalise_event_ids is not None:
        _logger.debug('Equalising classes %s for subject %s (%s).', equalise_event_ids, str(subj_id), subj_path.absolute())
        subj_epochs.equalize_event_counts(event_ids=equalise_event_ids)

    # Generate and add epoch subject id.
    if inc_subj_info_id:
        _logger.debug('Adding subject ID info for subject %s (%s).', str(subj_id), subj_path.absolute())
        subj_info = {'id': int(subj_id)}
        subj_epochs.info['subject_info'] = subj_info

    return subj_epochs


def get_proj_epochs(subj_ids, preload=True, equalise_event_ids=None, inc_subj_info_id=True):

    """
    Load and instantiate Epochs instances for a collection of subjects.
    :param subj_ids: list: the collection of int subject IDs that are to be loaded.
    :param preload: bool: whether the epochs should be loaded directly into memory or load them only when required.
    :param equalise_event_ids: list: contains the epoch classes that should be equalised with respect to time.
    :param inc_subj_info_id: bool: whether additional subject ID data is to be generated and appended to the loaded
        Epochs instance.
    :return: dict: containing loaded Epochs i.e. ID-Epochs mapping.
    """

    # Check param validity.
    if subj_ids is None:
        return None
    elif not subj_ids:
        return {}

    # Read all of the resource subject epochs.
    proj_epochs = {}
    for subj_id in subj_ids:
        subj_epochs = get_subj_epochs(subj_id=subj_id, preload=preload,
                                      equalise_event_ids=equalise_event_ids,
                                      inc_subj_info_id=inc_subj_info_id)
        if subj_epochs is None:
            _logger.warning('Subject %s Epochs is None.', str(subj_id))
            continue
        proj_epochs[subj_id] = subj_epochs
        _logger.debug('Subject %s Epochs loaded and added.', str(subj_id))

    return proj_epochs


def concat_epochs(epochs, add_offset=False, equalise_event_ids=None):

    # Copy all subjects (to avoid side effects); concat into one epoch.
    copied_epochs = [epoch.copy() for epoch in epochs]
    concat = mne.concatenate_epochs(epochs_list=copied_epochs, add_offset=add_offset)

    # The class IDs that are to be equalised.
    if equalise_event_ids is not None:
        concat.equalize_event_counts(event_ids=equalise_event_ids)

    return concat


def get_neuroscan_montage(azim=False, as_np_arr=True):

    # Get the location to the neuroscan montage path.
    neuroscan_path = pathlib.Path(PROJ_MONTAGES_DIR_PATH.joinpath('neuroscan_montage.mat'))
    if not neuroscan_path.exists() or not neuroscan_path.is_file() or neuroscan_path.is_dir():
        return None

    # Get the 3D coordinates within the mat file.
    neuroscan_items = get_mat_items(mat_path=neuroscan_path, mat_keys=['A'])
    if neuroscan_items is None:
        return None

    # Get the neuroscan 3d coords.
    neuroscan_3d_coords = neuroscan_items['A']
    if not azim:
        return np.asarray(neuroscan_3d_coords) if as_np_arr else neuroscan_3d_coords

    # Convert the 3D coords into 2D coords and return.
    neuroscan_2d_coords = []
    for coord in neuroscan_3d_coords:
        neuroscan_2d_coords.append(eeg_utils.azim_proj(coord))

    return np.asarray(neuroscan_2d_coords) if as_np_arr else neuroscan_3d_coords


def get_mne_montage(kind):
    if kind is not None:
        return mne.channels.make_standard_montage(kind=kind)
    return None


def set_epochs_mne_montage(epochs, kind, new=False):

    # Check param validity.
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


def get_epochs_labels(epochs):
    if epochs is not None:
        return epochs.events[:, 2]
    return None
