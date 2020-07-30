import EEGLearn.utils as eeg_utils
import EEGLearn.train as eeg_train

import logging

import mne
import tensorflow as tf
import numpy as np

from basicmi import montages, subjects, utils, features, train


_logger = utils.create_logger(name=__name__, level=logging.DEBUG)


def main():

    # Init library pseudo random number generators.
    pseudo_random_seed = 2435
    np.random.seed(pseudo_random_seed)
    tf.set_random_seed(pseudo_random_seed)

    # Read and load the electrode cap locations.
    electrode_locations = montages.get_neuroscan_montage(apply_azim=True)

    # Attributes defining what data should be loaded.
    load_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    equalise_event_ids = ['Left', 'Right', 'Bimanual']

    # Read and load epochs that we are concerned with.
    epochs = subjects.get_epochs(subject_ids=load_subjects, preload=True, equalise_event_ids=equalise_event_ids,
                                 add_subject_id_info=True)

    # Generate leave one out cross validation fold pairs based on the loaded epochs.
    trial_ids, trial_labels, fold_pairs = subjects.get_loocv_fold_pairs(epochs=epochs)

    # Labels must start from 0 (since for us they start at 1, sub 1 from each label.)
    trial_labels = np.array([label-1 for label in trial_labels])

    # The bands and windows defining the features that are to be extracted.
    bands = [(4, 8), (8, 13), (13, 30)]
    windows = utils.generate_windows(start=0, stop=5, step=5)

    # Generate PSD features for all of the loaded epochs; then concatenate them into one.
    epoch_feats = features.get_psd_features(epochs=epochs, windows=windows, bands=bands)
    feats = features.concat_features(windows=len(windows), epoch_feats=epoch_feats)

    # Generate images based on the generated features.
    images = utils.generate_images(features=feats, electrode_locations=electrode_locations, n_grid_points=32,
                                   normalise=True, edgeless=True)

    # Finally, run the classifier on the generated images.
    train.train_model(images=images, labels=trial_labels, folds=fold_pairs, model_type='cnn', batch_size=32,
                      num_epochs=30)


if __name__ == '__main__':
    main()
