import pathlib

import scipy.io as sio

from basicmi import utils

# Test function for obtaining the root path.
root_path = utils.get_proj_path()
print(root_path)
print('*-' * 50)

# Test function to get data dir.
data_path = utils.get_proj_sub_path('data')
print(data_path)
print('*-' * 50)

# Test matching the subject directories.
subject_paths = utils.get_matched_paths(data_path, pat='S*')
print(subject_paths)
print('*-' * 50)

# Test matching the session directories.
session_paths = utils.get_matched_paths(data_path, pat='S*/[Session|adjusted]*')
print(session_paths)
print('*-' * 50)

# Test the ability to remove bad paths.
for session_path in session_paths:
    bss_path = pathlib.Path(session_path.joinpath('brainstormstudy.mat'))
    bss_mat = sio.loadmat(bss_path)
    print(bss_mat.get('BadTrials'))
