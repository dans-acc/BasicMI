import logging
import pathlib

from typing import List, Dict, Tuple, Union

import mne
import numpy as np

from basicmi import utils


_logger = utils.create_logger(name=__name__, level=logging.DEBUG)


def get_subjects_epochs(dataset: str, subject_id: int, preload: bool = True, equalise_event_ids: List[str] = None,
                        add_subject_id_info: bool = True, drop_labels: List[int] = None) -> mne.Epochs:

    _logger.info('Loading Epochs for subject %d.', subject_id)

    # Read the subjects epoch data.
    subjects_path = pathlib.Path(pathlib.Path(__file__).parent
                                 .joinpath('resources//subjects//%s//S%d//epochs_epo.fif' % (dataset, subject_id)))
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

    # Drop trials that are not within the list.
    if drop_labels is not None:
        drop_epochs_trails_by_labels(epochs=subjects_epochs, drop_labels=drop_labels)

    return subjects_epochs


def get_epochs(dataset: str, subject_ids: List[int], preload: bool = True, equalise_event_ids: List[str] = None,
               add_subject_id_info: bool = True, drop_labels: List[int] = None) -> Dict[int, mne.Epochs]:

    # Dictionary mapping subject ids to the loaded epochs.
    epochs = {}

    # For each subject, load its epoch and store within the dictionary.
    for subject_id in np.sort(np.unique(subject_ids)):
        subjects_epochs = get_subjects_epochs(dataset=dataset, subject_id=subject_id, preload=preload,
                                              equalise_event_ids=equalise_event_ids,
                                              add_subject_id_info=add_subject_id_info,
                                              drop_labels=drop_labels)

        # Store the loaded epochs under the subject key.
        epochs[subject_id] = subjects_epochs
        _logger.debug('Added Epochs for subject %d.', subject_id)

    return epochs


def drop_epochs_trails_by_labels(epochs: mne.Epochs, drop_labels: List[int]):

    # Get the list of labels associated with the trials.
    trial_labels = get_epochs_trial_labels(epochs=epochs)

    # Loop through all trials, if their label is to be dropped storing their indices.
    drop_trial_indices = []
    for i in range(len(trial_labels)):
        if trial_labels[i] in drop_labels:
            _logger.debug('Added trial index %d to be dropped (label: %d)', i, trial_labels[i])
            drop_trial_indices.append(i)

    # Drop all trials based on index.
    _logger.debug('Number of labels before dropping %s is %d', str(drop_labels), len(trial_labels))
    epochs.drop(indices=drop_trial_indices)

    # Check that the number of trials and labels has decreased.
    trial_labels = get_epochs_trial_labels(epochs=epochs)
    _logger.debug('Number of labels after dropping %s is %d', str(drop_labels), len(trial_labels))


def remap_trail_labels(labels: Union[List, np.ndarray], new_labels: Dict[int, int]):

    # Loop through all of the labels. If a 2014 label exists, change it.
    for i in range(len(labels)):
        if labels[i] in new_labels:

            # Change the 2020 label to the 2014 label.
            _logger.debug('Changing %d at %d to %d', labels[i], i, new_labels[labels[i]])
            labels[i] = new_labels[labels[i]]


def get_epochs_trial_labels(epochs: mne.Epochs):

    # Get the labels associated with the trials of an epoch.
    return epochs.events[:, 2]


def get_trial_labels(epochs: Dict[int, mne.Epochs]) -> np.ndarray:

    # A concatenated list of sorted epochs trial labels.
    labels = []

    # Loop through all epochs ids, concatenating all trial labels to the list.
    unique_subject_ids = np.sort(np.unique(list(epochs.keys())))
    for subject_id in unique_subject_ids:

        # Because this operates on a sorted list, all Epochs must be present.
        if epochs[subject_id] is None:
            raise ValueError('Epochs for subject %d is None.' % subject_id)

        # Get the labels for the individual epochs; these are to be concatenated together for one big list.
        subject_epochs = epochs[subject_id]
        subject_epochs_labels = get_epochs_trial_labels(epochs=subject_epochs)

        # Concatenate the subjects trial labels (classes) to the bigger list.
        labels.extend(subject_epochs_labels)
        _logger.debug('%d trial labels for subject %d concatenated.', len(subject_epochs_labels), subject_id)

    return np.array(labels)


def get_epochs_trial_ids(subject_id: int, epochs: mne.Epochs) -> np.ndarray:

    # Generates a list of ids that match the number of trials for the epoch.
    ids = [subject_id for i in range(len(epochs))]
    return np.asarray(ids)


def get_trial_ids(epochs: Dict[int, mne.Epochs]) -> np.ndarray:

    # A concatenated list of sorted epoch ids (* number of trials).
    ids = []

    # Loop through all subjects concatenating their list of ids to the overall list.
    unique_subject_ids = np.sort(np.unique(list(epochs.keys())))

    for subject_id in unique_subject_ids:

        # Because we are operating on a sorted list, all epochs must be present.
        if epochs[subject_id] is None:
            raise ValueError('Epochs for subject %d is None.' % subject_id)

        # Get the ids for the individual epochs; there are to be concatenated together for one big list.
        subject_epochs = epochs[subject_id]
        subject_epochs_trials_ids = get_epochs_trial_ids(subject_id=subject_id, epochs=subject_epochs)

        # Concatenate the subjects trial ids to the bigger list.
        ids.extend(subject_epochs_trials_ids)
        _logger.debug('%d trial IDS for subject %d concatenated.', len(subject_epochs_trials_ids), subject_id)

    return np.array(ids)


def get_loocv_fold_pairs(epochs: Dict[int, mne.Epochs]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    _logger.info('Generating leave one out cross validation folds for subjects %s', len(list(epochs.keys())))

    # Get all of the IDs and labels for all trials within the epochs.
    ids = get_trial_ids(epochs=epochs)
    labels = get_trial_labels(epochs=epochs)

    # By definition, the two lists should match.
    assert len(labels) == len(ids)

    _logger.debug('List of IDs (%d) and labels (%d) generated for all subject Epochs.', len(ids), len(labels))

    # Leave one out cross validation pairs containing indices.
    folds = []

    # Generate a list of folds representing the training and test sets.
    unique_subject_ids = np.sort(np.unique(list(epochs.keys())))
    for unique_subject_id in unique_subject_ids:

        # Generate test (selected ids) and training (not selected ids) sets based on the selected sid samples.
        selected_ids = ids == unique_subject_id
        training_set_indices = np.squeeze(np.nonzero(np.bitwise_not(selected_ids)))
        test_set_indices = np.squeeze(np.nonzero(selected_ids))

        _logger.debug('For subject %d (with %d trials) training set is %d and test set is %d in length.',
                      unique_subject_id, len(epochs[unique_subject_id]), len(training_set_indices),
                      len(test_set_indices))

        # Shuffle only the the index values within each respective set.
        np.random.shuffle(training_set_indices)
        np.random.shuffle(test_set_indices)

        # Add the pairs to the list of folds.
        folds.append((np.array(training_set_indices), np.array(test_set_indices)))

    return ids, labels, np.asarray(folds)


