import EEGLearn.utils as eeg_utils
import EEGLearn.train as eeg_train

import logging

import mne
import tensorflow as tf
import numpy as np

from basicmi import montages, subjects, utils, features


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

    # The bands and windows defining the features that are to be extracted.
    bands = [(4, 8), (8, 13), (13, 30)]
    windows = utils.generate_windows(start=0, stop=5, step=0.5)

    # Generate PSD features for all of the loaded epochs; then concatenate them into one.
    epoch_feats = features.get_psd_features(epochs=epochs, windows=windows, bands=bands)
    feats = features.concat_features(windows=len(windows), epoch_feats=epoch_feats)

    # Generate the images that are to be used for training the models.



if __name__ == '__main__':
    main()


    """
    # The list of subjects used for training.
    subjects = [1, 2, 3, 4, 5, 6]

    # Init lib pseudo-random generators.
    np.random.seed(2435)
    tf.set_random_seed(2435)

    # Load the electrode cap, implicitly projecting 3D positions to 2D.
    neuroscan_coords = utils.get_neuroscan_montage(azim=True, as_np_arr=True)
    if neuroscan_coords is None:
        return

    # Load all subject mat files, creating corresponding Epochs instances for each.
    subject_epochs = utils.get_epochs(subj_ids=subjects, equalise_event_ids=['Left', 'Right', 'Bimanual'],
                                      inc_subj_info_id=True)
    if subject_epochs is None or not subject_epochs.keys():
        return

    # The windows and the frequency bands used to generate the images.
    windows = utils.generate_windows(0, 5, 0.5)
    frequency_bands = [(4, 8), (8, 13), (13, 30)]

    # Generate the windows for each of the epochs.
    subject_images = {}
    for subject_id, subject_epochs in subject_epochs.items():
        images = []
        for t_min, t_max in windows:
            subject_feats = features.get_subject_psd_feats(subject_epochs=subject_epochs, t_min=t_min, t_max=t_max,
                                                           freq_bands=frequency_bands, n_jobs=10, append_classes=False,
                                                           as_np_arr=True)
            img = gen_subj_imgs(subj_feats=subject_feats, cap_coords=neuroscan_coords, n_grid_points=32, normalise=True,
                                edgeless=False)
            images.append(img)
        images = np.array(images)
        subject_images[subject_id] = images
        print('Generated images for subject %d. Shape: %s', subject_id, str(images.shape))

    subject_feats = feats.get_psds_feats(epochs=subject_epochs, t_min=0, t_max=5, freq_bands=frequency_bands,
                                             n_jobs=3, append_classes=False, as_np_arr=True)
    if subject_feats is None or not subject_feats:
        return

    # Generate a list of windows for the epochs.

    # Generate images from the feature maps.
    proj_imgs = gen_proj_imgs(proj_ids=subjects,
                              proj_feats=subject_feats,
                              cap_coords=neuroscan_coords,
                              n_grid_points=32,
                              normalise=True,
                              edgeless=False)
    if proj_imgs is None:
        return
    

    # Generate the fold pairs according to leave-one-out validation.
    ids, samples, labels, folds = unpacked_folds(proj_ids=subjects,
                                                 proj_epochs=subject_epochs,
                                                 proj_feats=proj_imgs,
                                                 as_np_arr=True)
    if folds is None:
        return
        
    # Convert some types to np array.
    samples = np.array(samples)

    # Solves the batching issue.
    samples = np.expand_dims(samples, axis=0)

    # Labels must apparently start from 0.
    labels = [label-1 for label in labels]

    labels = np.array(labels)
    folds = np.array(folds)

    # Train a basic, single-frame CNN.
    cnn_accuracy = []
    for i in range(len(folds)):
        cnn_accuracy.append(eeg_train.train(images=samples,
                                            labels=labels,
                                            fold=folds[i],
                                            model_type='cnn',
                                            batch_size=32,
                                            num_epochs=60))
        tf.reset_default_graph()
    """
