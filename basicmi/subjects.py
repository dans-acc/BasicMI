import logging
import pathlib

from typing import List, Dict

import mne
import numpy as np

from basicmi import utils


_logger = utils.create_logger(name=__name__, level=logging.DEBUG)


def get_subjects_epochs(subject_id: int, preload: bool = True, equalise_event_ids: List[str] = None,
                        add_subject_id_info: bool = True) -> mne.Epochs:

    _logger.info('Loading Epochs for subject %d', subject_id)

    # Read the subjects epoch data.
    subjects_path = pathlib.Path(pathlib.Path(__file__).parent
                                 .joinpath('resources//subjects//S%d//epochs_epo.fif' % subject_id))
    subjects_epochs = mne.read_epochs(fname=str(subjects_path.absolute()), preload=preload)

    _logger.info('Epochs loaded for subject %d.', subject_id)

    # If not none, equalise the ids (classes) with respect to time.
    if equalise_event_ids is not None:
        _logger.debug('Equalising event IDs %s for subject %d.', str(equalise_event_ids), subject_id)
        subjects_epochs.equalize_event_counts(event_ids=equalise_event_ids)

    # Generate and add the subject ID to the Epochs info.
    if add_subject_id_info:
        _logger.debug('Adding ID info for subject %d.', subject_id)
        subject_id_info = {'id': int(subject_id)}
        subjects_epochs.info['subject_info'] = subject_id_info

    return subjects_epochs


def get_epochs(subject_ids: List[int], preload: bool = True, equalise_event_ids: List[str] = None,
               add_subject_id_info: bool = True) -> Dict[int, mne.Epochs]:
    epochs = {}
    for subject_id in np.sort(np.unique(subject_ids)):
        subjects_epochs = get_subjects_epochs(subject_id=subject_id, preload=preload,
                                              equalise_event_ids=equalise_event_ids,
                                              add_subject_id_info=add_subject_id_info)
        epochs[subject_id] = subjects_epochs
        _logger.debug('Added Epochs for subject %d.', subject_id)
    return epochs


def get_epochs_labels(epochs: mne.Epochs):
    return epochs.events[:, 2]
