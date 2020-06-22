import pathlib

from basicmi import dataset
from basicmi import tools

# Get a random session from listed paths.
session = dataset.SESSION_DIR_PATHS[0]
print(session)

dataset.get_session_epochs(session_path=session,
                           epoch_class='*Left*')

"""

# Get the list of bad file names associated with the session.
bss_items = tools.get_mat_items(mat_path=pathlib.Path(session.joinpath('brainstormstudy.mat')),
                                mat_keys=['BadTrials'])
print(bss_items)
print(type(bss_items['BadTrials']))

# Get a class from the data set.
lefts = tools.get_dir_paths(dir_path=session,
                            regex='*Left*',
                            only_files=True,
                            bad_files=bss_items['BadTrials'])

# Print the lefts.
for left in lefts:
    print(left)
    
"""

"""
Filter the epochs appropriately;
Recreate the models.
Filter the epochs using 'improved' technique.
Recreate the models.
Generate phase locking values (?)
Recreate the models.
Create the CNN for the dataset.
"""


"""
for session in SESSION_DIR_PATHS:
    get_session_epochs(session)

test_paths = get_dir_paths(ROOT_DIR_PATH, bad_files=['README.md', 'setup.py'])
for tp in test_paths:
    print(tp)
"""

"""
for session_path in SESSION_DIR_PATHS:

    lefts = get_dir_paths(session_path, regex='*Left*', only_files=True)
    for left in lefts:
        items = get_mat_items(left, ['F'])
        epochs = np.array(items['F'])
        epochs_df = pd.DataFrame(epochs).transpose()
        epochs_df.plot()
        print((epochs_df))
"""

"""
import os
import json

import pathlib

import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib as plt

from basicmi import tools

print('*' * 100)
paths = tools.get_session_dirs()
print(paths)
print('*' * 100)

for session in paths:
    classes = tools.get_class_mats(session)
    print(classes)


# Get the data directory path.
FILE_PATH = pathlib.Path(__file__)
DATA_DIR_PATH = pathlib.Path(FILE_PATH.parent.parent.absolute().joinpath('data'))

# Test reading the mat files.
mat_file_path = pathlib.Path(DATA_DIR_PATH).joinpath('S1/Session1_MI_13_11_2019_15_18_24_0000_interpbad_concat/data_Bimanual_trial001_montage.mat')
print('The name of the file is: %s' % mat_file_path.name)
mat = sio.loadmat(mat_file_path)

# Print the mat file.
print(mat)
print('-' * 100)
print(mat.keys())
print('-' * 100)
print(mat.items())

print(mat.get('Name'))
"""