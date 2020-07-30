import logging

from typing import List, Dict, Tuple

import mne
import numpy as np
import tensorflow as tf
import EEGLearn.train as eegl_train

from basicmi import utils


_logger = utils.create_logger(name=__name__, level=logging.DEBUG)


def train_model(images: np.ndarray, labels: np.ndarray, folds: np.ndarray, model_type: str = 'cnn',
                batch_size: int = 32, num_epochs: int = 60) -> List:

    results = []

    # For each of the folds train the model.
    for i in range(len(folds)):

        _logger.info('Training model for fold: %d.', i)

        # Train the model and store the results.
        fold_result = eegl_train.train(images=images, labels=labels, fold=folds[i], model_type=model_type,
                                       batch_size=batch_size, num_epochs=num_epochs)
        results.append(fold_result)
        _logger.info('Fold %d result results: %s.', i, results)

        # Reset the graph (otherwise there is an error that occurs).
        tf.reset_default_graph()

    return results
