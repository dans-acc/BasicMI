import pathlib


def get_basicmi_dir():

    # Get the path to the tools file.
    path = pathlib.Path(__file__)
    if path is None:
        return None
    return path.parent.parent


def get_data_dir():

    # Get the path to the root directory of the project.
    basicmi_dir_path = get_basicmi_dir()
    if basicmi_dir_path is None:
        return None
    return pathlib.Path(basicmi_dir_path.joinpath('data'))


def get_session_dirs(session_pat='S*/[adjusted|Session]*'):

    if session_pat is None:
        return None

    # Get the path to the project session directory.
    data_dir_path = get_data_dir()
    if data_dir_path is None or not data_dir_path.is_dir():
        return None

    # Get the session paths for each of the subjects.
    paths = [session_dir for session_dir in data_dir_path.glob(session_pat)]
    return paths


def get_class_mats(dir_path, bad_mats=None, class_pat='*[Bimanual|Left|Right]*'):

    # Check that paths are valid.
    if dir_path is None or class_pat is None:
        return

    # Get the class paths to the mat files.
    class_mat_paths = [class_mat for class_mat in dir_path.glob(class_pat)]

    # Remove the bad files.
    if bad_mats is not None:
        for i in range(len(class_mat_paths) - 1, -1, -1):
            class_mat_path = class_mat_paths[i]
            if class_mat_path.name in bad_mats:
                del class_mat_paths[i]

    # Finally, return all the class mat files.
    return class_mat_paths


def get_epochs(mat=None):

    # The mat file cannot be none.
    if mat is None:
        return None

    # Get the associated epochs.
    pass


