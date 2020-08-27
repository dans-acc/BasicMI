import logging

from typing import List, Dict, Tuple

import mne
import numpy as np
import tensorflow as tf
import EEGLearn.train as eegl_train

from basicmi import utils


_logger = utils.create_logger(name=__name__, level=logging.DEBUG)


def train_eegl_model(images: np.ndarray, labels: np.ndarray, folds: np.ndarray, model_type: str = 'cnn',
                     batch_size: int = 32, num_epochs: int = 60, reuse_cnn: bool = False,
                     dropout_rate: float = 0.5, learning_rate: float = 1e-4, learning_rate_default=1e-4, weight_decay: float = 1e-3,
                     decay_rate: float = 0.75) -> List:

    # Fixes the issue surrounding wrong log paths.
    eegl_train.model_type = model_type

    results = []

    # For each of the folds train the model.
    for i in range(len(folds)):

        _logger.info('Training model for fold: %d.', i)

        # Train the model and store the results.
        fold_result = eegl_train.train(images=images, labels=labels, subj_id=i, fold=folds[i], model_type=model_type,
                                       batch_size=batch_size, num_epochs=num_epochs, reuse_cnn=reuse_cnn,
                                       dropout_rate=dropout_rate, learning_rate=learning_rate,
                                       learning_rate_default=learning_rate_default, weight_decay=weight_decay,
                                       decay_rate=decay_rate, image_size=32)

        results.append(fold_result)
        _logger.info('Fold %d result results: %s.', i, results)

        # Reset the graph (otherwise there is an error that occurs).
        tf.reset_default_graph()

    # Print the results for each subject.
    for i in range(len(results)):
        _logger.info('Last train accuracy: %.16f; best validation accuracy: %.16f; test accuracy: %.16f; '
                     'last validation accuracy: %.16f; last test accuracy: %.16f',
                     results[i][0],
                     results[i][1],
                     results[i][2],
                     results[i][3],
                     results[i][4])

    # Extract the validation and test accuracies.
    last_training_accuracies = []
    best_validation_accuracies = []
    test_accuracies = []
    last_validation_accuracies = []
    last_test_accuracies = []

    # Populate the lists for obtaining mean and std etc.
    for fold_result in results:
        last_training_accuracies.append(fold_result[0])
        best_validation_accuracies.append(fold_result[1])
        test_accuracies.append(fold_result[2])
        last_validation_accuracies.append(fold_result[3])
        last_test_accuracies.append(fold_result[4])

    # Summarise the last training accuracies.
    _logger.info('Mean last training accuracy: %.16f; last training accuracy std: %.16f',
                 np.mean(last_training_accuracies),
                 np.std(last_training_accuracies))

    # Summarise the best validation accuracies.
    _logger.info('Mean best validation accuracy: %.16f; best validation accuracy std: %.16f',
                 np.mean(best_validation_accuracies),
                 np.std(best_validation_accuracies))

    # Summarise the test accuracies.
    _logger.info('Mean test accuracy: %.16f; test accuracy std: %.16f',
                 np.mean(test_accuracies),
                 np.std(test_accuracies))

    # Summarise the last validation accuracies.
    _logger.info('Mean last validation accuracy: %.16f; last validation accuracy std: %.16f',
                 np.mean(last_validation_accuracies),
                 np.std(last_validation_accuracies))

    # Summarise the last test accuracies.
    _logger.info('Mean last test accuracy: %.16f; last test accuracy std: %.16f',
                 np.mean(last_test_accuracies),
                 np.std(last_test_accuracies))

    return results
