from basicmi import tools


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