import pathlib
import scipy.io as sio


def get_proj_path():

    # Get the project path relative to this file.
    path = pathlib.Path(__file__)
    if path is None:
        return
    return pathlib.Path(path.parent.parent)


def get_proj_sub_path(sub_path='basicmi'):

    # Get a path relative to the projects root directory.
    proj_path = get_proj_path()
    if proj_path is None:
        return
    return pathlib.Path(proj_path.joinpath(sub_path))


def get_matching_paths(from_path, pattern):

    # Check that the parameters are valid.
    if from_path is None or pattern is None:
        return None
    elif not from_path.is_dir():
        return None

    # Match all sub paths.
    paths = [pathlib.Path(path) for path in from_path.glob(pattern)]
    return paths


def get_mat_items(mat_path, mat_keys=None):

    # Check that the mat file exists.
    if mat_path is None or not mat_path.exists():
        return None

    # Read the mat file, if no keys, return the entire dict.
    mat_file = sio.loadmat(mat_path)
    if mat_keys is None:
        return mat_file
    else:

        # Only get a subset of the mat keys.
        return {key: mat_file.get(key) for key in mat_keys}


def get_dir_paths(dir_path, regex='*', only_files=False, bad_files=None):

    # Check that the dir path is valid.
    if dir_path is None or not dir_path.is_dir():
        raise NotADirectoryError('dir_path is None or not a directory.')

    # Read all paths within the directory.
    paths = dir_path.glob(regex)
    if only_files:
        paths = [path for path in paths if path.is_file() and not path.is_dir()]

    # If no bad files are defined, return all paths.
    if bad_files is None or not bad_files.any():
        return paths
    else:

        # Filter out all of the bad files from the data set.
        return [path for path in paths if path.name not in bad_files]


# Common paths for the project.
ROOT_DIR_PATH = get_proj_path()
DATA_DIR_PATH = get_proj_sub_path('data')