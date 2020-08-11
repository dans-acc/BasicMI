import EEGLearn.utils as eeg_utils
import EEGLearn.train as eeg_train

import logging
import os
import pathlib

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
    drop_labels = [2]
    equalise_event_ids = ['Left', 'Right', 'Bimanual']

    # Read and load epochs that we are concerned with.
    epochs = subjects.get_epochs(subject_ids=load_subjects, preload=True, equalise_event_ids=equalise_event_ids,
                                 add_subject_id_info=True, drop_labels=drop_labels)

    # Generate leave one out cross validation fold pairs based on the loaded epochs.
    trial_ids, trial_labels, fold_pairs = subjects.get_loocv_fold_pairs(epochs=epochs)

    # Since class labels must start at 1; remap the values such that the meet the latter requirement.
    subjects.remap_trail_labels(trial_labels, new_labels={1: 0, 2: 1, 3: 2})

    # The bands and windows defining the features that are to be extracted.
    bands = [(4, 8), (8, 12), (12, 30)]
    windows = utils.generate_windows(start=-2, stop=5, step=1)

    # Attributes that help uniquely identify files where features are held.
    file_name_attributes = {'Subj': load_subjects, 'Equalised': equalise_event_ids, 'Drop': drop_labels,
                            'Bands': bands, 'Win': windows}

    # Path where previously generated features are to be found.
    epoch_feats_path = pathlib.Path(pathlib.Path(__file__).parent.joinpath(
        'features//psd//%s.mat' % str(file_name_attributes).replace(' ', '')))

    # Determine if the features have already been generated.
    if epoch_feats_path.exists():

        _logger.info('Loading PSD features...')

        # Load the previously saved features; optimises runs because they do not need to be generated each time.
        feats = utils.read_mat_items(mat_path=epoch_feats_path, mat_keys=['feats'])['feats']
    else:

        _logger.info('Generating PSD features...')

        # Generate features for all of the loaded epochs; concatenate them into one (indexed by windows); then save them
        epoch_feats = features.get_psd_features(epochs=epochs, windows=windows, bands=bands)
        feats = features.concat_features(windows=len(windows), epoch_feats=epoch_feats)
        utils.save_mat_items(mat_path=epoch_feats_path, mat_items={'feats': feats})

    # Attributes defining image properties.
    n_image_grid_points = 32
    normalise_image = True
    edgeless_image = True

    # Append additional file name attributes enable the image to be identified.
    file_name_attributes['points'] = n_image_grid_points
    file_name_attributes['normalise'] = normalise_image
    file_name_attributes['edgeless'] = edgeless_image

    # The path where previously generates images are to be found.
    feats_images_path = pathlib.Path(pathlib.Path(__file__).parent.joinpath(
        'images//psd//%s.mat' % str(file_name_attributes).replace(' ', '')))

    # Determine if the images have already been generated for the features.
    if feats_images_path.exists():

        _logger.info('Loading images for PSD features...')

        # Load the previously saved images; optimises runs because they do not need to be generated each time.
        images = utils.read_mat_items(mat_path=feats_images_path, mat_keys=['images'])['images']
    else:

        _logger.info('Generating images for PSD features...')

        # Generate images based on the generated features.
        images = utils.generate_images(features=feats, electrode_locations=electrode_locations,
                                       n_grid_points=n_image_grid_points, normalise=normalise_image,
                                       edgeless=edgeless_image)

        # Save the images, preventing the need to generate them again.
        utils.save_mat_items(mat_path=feats_images_path, mat_items={'images': images})

    # Finally, run the classifier on the generated images.
    train.train_eegl_model(images=images, labels=trial_labels, folds=fold_pairs, model_type='lstm', batch_size=32,
                           num_epochs=10, reuse_cnn=False, dropout_rate=0.5, learning_rate_default=1e-3)


if __name__ == '__main__':

    # Set tensorflow attributes.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Run the classification algorithm.
    main()
