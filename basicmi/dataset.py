import pathlib

import pandas as pd

from basicmi import tools


# Paths for the subjects and sessions.
SUBJECT_DIR_PATHS = tools.get_matching_paths(from_path=tools.DATA_DIR_PATH, pattern='S*')
SESSION_DIR_PATHS = tools.get_matching_paths(from_path=tools.DATA_DIR_PATH, pattern='S*/[Session|adjusted]*')


def get_session_epochs(session_path, epoch_class='*', transpose=True):

    # Check that the session path is valid.
    if session_path is None or not session_path.is_dir():
        raise NotADirectoryError('Session path is not a directory.')

    # Get the brain storm study for bad trials.
    bss_items = tools.get_mat_items(mat_path=pathlib.Path(session_path.joinpath('brainstormstudy.mat')),
                                    mat_keys=['BadTrials'])
    if bss_items is None:
        raise FileNotFoundError('Failed to read brain storm study file')

    # Get a class from the dataset.
    epoch_class_paths = tools.get_dir_paths(dir_path=session_path,
                                            regex=epoch_class,
                                            only_files=True,
                                            bad_files=bss_items['BadTrials'])

    # Get the epochs from the dataset.
    for epoch_path in epoch_class_paths:
        trial_items = tools.get_mat_items(mat_path=epoch_path,
                                    mat_keys=['F'])
        trial_df = pd.DataFrame(trial_items['F']).transpose()
        print(trial_df)


def get_all_epochs(session_paths=SESSION_DIR_PATHS, epoch_class='*', transpose=False):

    # Make some checks.
    if session_paths is None or not session_paths:
        raise ValueError('Invalid session paths.')

    for session_path in session_paths:
        get_session_epochs(session_path=session_path,
                           epoch_class=epoch_class,
                           transpose=transpose)