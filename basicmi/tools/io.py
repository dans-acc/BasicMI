import pathlib
import scipy


def get_proj_path():

    # Get the path to this file.
    path = pathlib.Path(__file__)
    if path is None:
        return None

    # Get the root directory of the project relative to this file.
    return path.parent.parent.parent


def get_proj_sub_path(sp='data'):

    # Get the projects root path.
    proj_path = get_proj_path()
    if proj_path is None or sp is None:
        return

    return proj_path.joinpath(sp)


def get_matched_sub_paths(root, pat='S*/[adjusted|Session]*'):

    # Check that the root from which we are matching is a directory.
    if root is None or not root.is_dir():
        return
    return [file for file in root.glob(pat)]