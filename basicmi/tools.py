import pathlib

import mne
import scipy.io as sio

# Commonly used paths relative to the tools.py file.
PROJ_DIR_PATH = pathlib.Path(pathlib.Path(__file__).parent.parent)
DATA_DIR_PATH = pathlib.Path(PROJ_DIR_PATH.joinpath('data'))
MI_DATA_DIR_PATH = pathlib.Path(DATA_DIR_PATH.joinpath('mi'))


def get_subj_epochs(subj_id, preload=True, equalise_events_ids=None):

    # Sanity checks.
    if subj_id is None:
        return None

    # Get the subject path.
    subj_path = pathlib.Path(MI_DATA_DIR_PATH.joinpath('S%d//epochs_epo.fif'))
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
        subj_epoch = get_subj_epochs(subj_id=subj_id, preload=preload,
                                     equalise_events_ids=equalise_events_ids)
        if subj_epoch is None:
            continue
        subj_epochs[subj_id] = subj_epoch

    return subj_epochs

