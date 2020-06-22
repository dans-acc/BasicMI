import pathlib

from basicmi import tools


# Paths for the subjects and sessions.
SUBJECT_DIR_PATHS = tools.get_matching_paths(from_path=tools.DATA_DIR_PATH, pattern='S*')
SESSION_DIR_PATHS = tools.get_matching_paths(from_path=tools.DATA_DIR_PATH, pattern='S*/[Session|adjusted]*')


def get_session_epochs(session_path, epoch_class='*'):

    # Check that the session path is valid.
    if session_path is None or not session_path.is_dir():
        raise NotADirectoryError('Param session_path must be a session directory.')

    # Get the list of bad file names.
    bss_items = tools.get_mat_items(mat_path=pathlib.Path(session_path.joinpath('brainstormstudy.mat')),
                                    mat_keys=['BadTrials'])
    if bss_items is None:
        raise ValueError('BadTrials not found for %s' % session_path.absolute())

    print(bss_items)

    # Get the list of good and valid class paths from within the session.
    class_paths = tools.get_dir_paths(dir_path=session_path,
                                      regex=epoch_class,
                                      only_files=True,
                                      bad_files=bss_items['BadTrials'])
    if class_paths is None:
        return ValueError('Unable to find class paths')

    # Get the epochs out of the data set.
    for path in class_paths:
        class_items = tools.get_mat_items(mat_path=path,
                                          mat_keys=['F'])
        print(class_items)


def get_all_epochs(session_paths=SESSION_DIR_PATHS, epoch_class=None):

    # Loop through each of the sessions, getting the epoch classes.
    for session_path in session_paths:
        get_session_epochs(session_path=session_path, epoch_class=epoch_class)

