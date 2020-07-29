import logging
import pathlib

from typing import List, Dict

import mne
import numpy as np
import EEGLearn.utils as eegl_utils

from basicmi import utils


_logger = utils.create_logger(name=__name__, level=logging.DEBUG)


def get_neuroscan_montage(apply_azim: bool = True) -> np.ndarray:

    _logger.debug('Loading Neuroscan montage.')

    # Read and load the locations for the Neuroscan cap from the mat file.
    neuroscan_path = pathlib.Path(pathlib.Path(__file__).parent.joinpath('resources//montages//neuroscan_montage.mat'))
    neuroscan_mat_items = utils.read_mat_items(mat_path=neuroscan_path, mat_keys=['A'])

    _logger.debug('Neuroscan mat items loaded.')

    # Get the 3D electrode locations; return if they are not being projected.
    neuroscan_3d_locs = neuroscan_mat_items['A']
    if not apply_azim:
        _logger.debug('Returning 3D Neuroscan electrode locations.')
        return np.array(neuroscan_3d_locs)

    # Convert the 3D coordinates into 2D locations.
    neuroscan_2d_locs = []
    for electrode_loc in neuroscan_3d_locs:
        neuroscan_2d_locs.append(eegl_utils.azim_proj(electrode_loc))

    _logger.debug('Returning 2D Neuroscan electrode locations.')
    return np.array(neuroscan_2d_locs)


def set_mne_montage(epochs: mne.Epochs, kind: str, new: bool = False) -> mne.Epochs:

    # Get the montage that is to be set.
    mne_montage = mne.channels.make_standard_montage(kind=kind)

    # Set the MNE montage.
    epochs = epochs.copy() if new else epochs
    epochs.set_montage(montage=mne_montage)

    _logger.debug('MNE montage successfully set for Epochs: %s.', str(epochs))

    return epochs


