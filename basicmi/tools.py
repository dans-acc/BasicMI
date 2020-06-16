import pathlib


def get_fp(file):

    # Get Path instance from any file.
    if file is None:
        return None
    return pathlib.Path(file)


def get_basicmi_dp():

    # Get basicmi project directory.
    path = get_fp(__file__)
    if path is None:
        return None

    # Return the root.
    return path.parent.parent

