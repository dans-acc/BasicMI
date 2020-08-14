import EEGLearn.utils as eeg_utils
import EEGLearn.train as eeg_train

import logging
import os
import pathlib

from typing import List, Dict, Tuple

import mne
import tensorflow as tf
import numpy as np

from basicmi import montages, subjects, utils, features, train


_logger = utils.create_logger(name=__name__, level=logging.DEBUG)


def load_or_generate_fif_feats(dataset: str = '2020', load_subjects: List[int] = None, drop_labels: List[int] = None,
                               equalise_event_ids: List[str] = None, new_trial_labels: Dict[int, int] = None,
                               bands: List[Tuple[int, int]] = None, windows: List[Tuple[float, float]] = None,
                               n_image_grid_points: int = 32, normalise_image: bool = True, edgeless_image: bool = True,
                               electrode_locations: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # If the dataset is none, assume the latest dataset is being used.
    if dataset is None:
        dataset = '2020'

    _logger.info('Loading or generating features and images using the .FIF and MNE (using the %s dataset)', dataset)

    # If the subjects that are to be loaded does not match, assume all are being loaded.
    if load_subjects is None:
        load_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # Read and load epochs that we are concerned with.
    epochs = subjects.get_epochs(dataset=dataset, subject_ids=load_subjects, preload=True,
                                 equalise_event_ids=equalise_event_ids, add_subject_id_info=True,
                                 drop_labels=drop_labels)

    # Generate leave one out cross validation fold pairs based on the loaded epochs.
    trial_ids, trial_labels, fold_pairs = subjects.get_loocv_fold_pairs(epochs=epochs)

    # If none, the default assumes all classes are present.
    if new_trial_labels is None:
        new_trial_labels = {1: 0, 2: 1, 3: 2}
        _logger.debug('New trial labels is None. Using default trial labels: %s', str(new_trial_labels))

    # Since class labels must start at 1; remap the values such that the meet the latter requirement.
    subjects.remap_trail_labels(trial_labels, new_labels=new_trial_labels)

    # The bands define the frequencies from which the features are to be extracted; if none, assumes defined.
    if bands is None:
        bands = [(4, 8), (8, 12), (12, 30)]

    # If there are no windows present, generate default ones.
    if windows is None:
        windows = utils.generate_windows(start=-2, stop=5, step=1)
        _logger.debug('Windows is None. Using default windows: %s', str(windows))

    try:

        # Generate a path to where the features are to be stored; paths are unique; if too long, we cannot save.
        epoch_feats_path = utils.get_dict_path(
            from_path=pathlib.Path(pathlib.Path(__file__).parent.joinpath('features//psd')),
            dictionary={
                'Dataset': dataset,
                'Subjects': load_subjects,
                'Equalised': equalise_event_ids,
                'Dropped': drop_labels,
                'Bands': bands,
                'Windows': windows
            }).joinpath('features.mat')

        # This call forces a filename check (to make sure its valid).
        epoch_feats_path.exists()

    except OSError:

        # Cannot save nor load the features (dictionary names are too long).
        _logger.debug('Cannot load nor save features - invalid path.')
        epoch_feats_path = None

    # Determine if the features have already been generated.
    if epoch_feats_path is not None and epoch_feats_path.exists():

        # Load the previously saved features; optimises runs because they do not need to be generated each time.
        _logger.info('Loading PSD features...')
        feats = utils.read_mat_items(mat_path=epoch_feats_path, mat_keys=['feats'])['feats']

    else:

        # Generate features for all of the loaded epochs; concatenate them into one (indexed by windows).
        _logger.info('Generating PSD features...')
        epoch_feats = features.get_psd_features(epochs=epochs, windows=windows, bands=bands)
        feats = features.concat_features(windows=len(windows), epoch_feats=epoch_feats)

        # Save the generated features if a valid mat file path exists.
        if epoch_feats_path is not None:
            utils.save_mat_items(mat_path=epoch_feats_path, mat_items={'feats': feats})

    _logger.debug('Features shape: %s', str(feats.shape))

    try:

        # Generate a 2014 path to where images are stored.
        feats_images_path = utils.get_dict_path(
            from_path=pathlib.Path(pathlib.Path(__file__).parent.joinpath('images//psd')),
            dictionary={
                'Dataset': dataset,
                'Subjects': load_subjects,
                'Equalised': equalise_event_ids,
                'Dropped': drop_labels,
                'Bands': bands,
                'Windows': windows,
                'Points': n_image_grid_points,
                'Normalise': normalise_image,
                'Edgeless': edgeless_image
            }).joinpath('images.mat')

        # This call forces a filepath check.
        feats_images_path.exists()

    except OSError:

        # Cannot load nor store images (dictionary names are too long).
        _logger.debug('Cannot load nor save images - invalid path.')
        feats_images_path = None

    # Determine if the images have already been generated for the features.
    if feats_images_path is not None and feats_images_path.exists():

        # Load the previously saved images; optimises runs because they do not need to be generated each time.
        _logger.info('Loading images for PSD features...')
        images = utils.read_mat_items(mat_path=feats_images_path, mat_keys=['images'])['images']

    else:

        # If the electrode locations have not already been loaded, load them.
        if electrode_locations is None:
            electrode_locations = montages.get_neuroscan_montage(apply_azim=True)

        # Generate images based on the features.
        _logger.info('Generating images for PSD features...')
        images = utils.generate_images(features=feats, electrode_locations=electrode_locations,
                                       n_grid_points=n_image_grid_points, normalise=normalise_image,
                                       edgeless=edgeless_image)

        # Save the images if a valid filepath exists; optimises successive runs as they do not need to be regenerated.
        if feats_images_path is not None:
            utils.save_mat_items(mat_path=feats_images_path, mat_items={'images': images})

    return images, trial_labels, fold_pairs


def load_or_generate_mat_feats():
    pass


def main():

    # Init library pseudo random number generators.
    pseudo_random_seed = 2018
    np.random.seed(pseudo_random_seed)
    tf.set_random_seed(pseudo_random_seed)

    # Read and load the electrode cap locations.
    electrode_locations = montages.get_neuroscan_montage(apply_azim=True)

    # Attributes defining what data should be loaded.
    dataset = '2020'
    load_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    drop_labels = None
    equalise_event_ids = None
    new_trial_labels = {1: 0, 2: 1, 3: 2}

    # The bands and windows defining the features that are to be extracted.
    bands = [(4, 8), (8, 12), (12, 30)]
    windows = utils.generate_windows(start=-2, stop=5, step=7)

    # Attributes defining image properties.
    n_image_grid_points = 32
    normalise_image = True
    edgeless_image = False

    # Generate the images, labels and folds associated with prior parameters.
    images, trial_labels, fold_pairs = load_or_generate_fif_feats(dataset=dataset, load_subjects=load_subjects,
                                                                  drop_labels=drop_labels,
                                                                  equalise_event_ids=equalise_event_ids,
                                                                  new_trial_labels=new_trial_labels, bands=bands,
                                                                  windows=windows,
                                                                  n_image_grid_points=n_image_grid_points,
                                                                  normalise_image=normalise_image,
                                                                  edgeless_image=edgeless_image,
                                                                  electrode_locations=electrode_locations)

    # Model parameters.
    model_type = 'cnn'
    reuse_cnn = False

    batch_size = 32
    num_epochs = 6000

    learning_rate_default = 1e-4
    learning_rate = 1e-4 / 32 * batch_size

    dropout_rate = 0.5
    decay_rate = 0.78
    weight_decay = 1e-4

    # Finally, run the classifier on the generated images.
    train.train_eegl_model(images=images, labels=trial_labels, folds=fold_pairs, model_type=model_type,
                           reuse_cnn=reuse_cnn, batch_size=batch_size, num_epochs=num_epochs,
                           learning_rate_default=learning_rate_default, learning_rate=learning_rate,
                           dropout_rate=dropout_rate, decay_rate=decay_rate, weight_decay=weight_decay)


if __name__ == '__main__':

    # Set tensorflow attributes.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Run the classification algorithm.
    main()
