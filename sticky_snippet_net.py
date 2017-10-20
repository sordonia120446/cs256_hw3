"""
Training in tensorflow.

:authors Jason, Nick, Sam
"""

import argparse

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def dnn_model_fn(features, labels, mode):

    # Load training data

    # Shape input layer
    l0 = None

    # Define hidden layers
    l1 = tf.layers.dense(inputs=l0, units=40, activation=tf.nn.relu)
    l2 = tf.layers.dense(inputs=l1, units=40, activation=tf.nn.relu)
    l3 = tf.layers.dense(inputs=l2, units=40, activation=tf.nn.relu)
    l4 = tf.layers.dense(inputs=l3, units=40, activation=tf.nn.relu)

    # Define output (logit) layer
    logits = tf.layers.dense(inputs=l4, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=logits
    )

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"]
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


def main(args):

    # Init estimator
    sticky_classifier = tf.estimator.Estimator(
        model_fn=dnn_model_fn,
        model_dir=args.model_file
    )

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    # TODO Train model

    # TODO Eval model


"""CLARGS"""

parser = argparse.ArgumentParser(
    description='Train and test the neural net',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

parser.add_argument(
    'mode',
    help='Specify whether to test or train as well as how to train'
)

parser.add_argument(
    'model_file',
    help='Path to the file containing the trained weights'
)

parser.add_argument(
    'data_folder',
    help='Path to the folder containing training/testing data'
)

if __name__ == '__main__':
    args = parser.parse_args()

    main(args)

