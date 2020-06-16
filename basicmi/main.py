import os
import json

import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib as plt

from basicmi import tools

print('*' * 100)
tools.get_session_dirs()
print('*' * 100)

# Get the data directory path.
#FILE_PATH = pathlib.Path(__file__)
#DATA_DIR_PATH = pathlib.Path(FILE_PATH.parent.parent.absolute().joinpath('data'))

# Test reading the mat files.
#mat_file_path = pathlib.Path(DATA_DIR_PATH).joinpath('S1/Session1_MI_13_11_2019_15_18_24_0000_interpbad_concat/data_Bimanual_trial001_montage.mat')
#mat = sio.loadmat(mat_file_path)

# Print the mat file.
#print(mat)
#print(mat.keys())
#print('-' * 100)
#print(mat.items())
