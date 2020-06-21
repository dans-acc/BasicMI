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


def get_dir_paths(dir_path, only_files=False, bad_files=None):

    # Check that the dir path is valid.
    if dir_path is None or not dir_path.is_dir():
        return None

    # Read all paths within the directory.
    paths = dir_path.glob('*')
    if only_files:
        paths = [path for path in paths if path.is_file() and not path.is_dir()]

    # If no bad files are defined, return all paths.
    if not bad_files or len(bad_files) == 0:
        return paths
    else:

        # Filter out all of the bad files from the data set.
        return [path for path in paths if path.name not in bad_files]


# Project directories.
ROOT_DIR_PATH = get_proj_path()
DATA_DIR_PATH = get_proj_sub_path('data')

test_paths = get_dir_paths(ROOT_DIR_PATH, bad_files=['README.md', 'setup.py'])
for tp in test_paths:
    print(tp)

# Session paths.
SUBJECT_DIR_PATHS = get_matching_paths(DATA_DIR_PATH, pattern='S*')
SESSION_DIR_PATHS = get_matching_paths(DATA_DIR_PATH, pattern='S*/[Session|adjusted]*')

for session_path in SESSION_DIR_PATHS:
    bss = get_matching_paths(session_path, pattern='brainstormstudy.mat')
    mat = get_mat_items(bss[0], ['BadTrials'])
    print(mat)

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