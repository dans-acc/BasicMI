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
        _logger.info('Best validation accuracy: %.16f; test accuracy: %.16f', results[i][1], results[i][2])

    # Extract the validation and test accuracies.
    validation_accuracies = []
    test_accuracies = []

    for fold_result in results:
        validation_accuracies.append(fold_result[1])
        test_accuracies.append(fold_result[2])

    # Print the final summary.
    _logger.info('Mean validation accuracy: %.16f; validation accuracy std: %.16f', np.mean(validation_accuracies),
                 np.std(validation_accuracies))

    _logger.info('Mean test accuracy: %.16f; test accuracy std: %.16f', np.mean(test_accuracies), np.std(test_accuracies))

    return results
