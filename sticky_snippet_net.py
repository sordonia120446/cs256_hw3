"""
Training in tensorflow.
Build a classifier for "alien DNA."  They have six possible labels:
    NONSTICK
    12-STICKY
    34-STICKY
    56-STICKY
    78-STICKY
    STICK_PALINDROME 

:authors Jason, Nick, Sam
"""

import argparse
import glob
import os

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)

label_map = {
    'NONSTICK': 1,
    '12-STICKY': 2,
    '34-STICKY': 3,
    '56-STICKY': 4,
    '78-STICKY': 5,
    'STICK_PALINDROME': 6
}

MINIBATCH_SIZE = 100

def convert_data(input_line):
    """
    Converts the data from raw DNA to ASCII char numpy array.

    :param input_line: line of input data (DNA & label)
    :returns type <numpy arr>: tf-friendly input vector
    """
    
    try:
        data, label = input_line.split(",")
    except ValueError:
        raise Exception('Check the input file format')

    data.strip()  # remove whitespace
    label.strip()  # remove whitespace

    # Convert letters into ASCII
    data = [ord(c) for c in data]

    # Map labels to ints
    try:
        label = label_map[label]
    except KeyError:
        raise Exception(f'Check spelling on label {label}')

    return np.array(data, dtype=np.float32), label


def load_data(data_folder):
    """
    Iterate through each file in data_folder and construct
    the input data features (40-len DNA).

    :param data_folder: path to folder of data
    :param exclude: kth file to exclude in cross-validation
    :returns type list: list of dicts for DNA-arr & label
    """
    
    data_files = os.path.join(data_folder, '*.txt')
    data = []
    for f_path in glob.glob(data_files):

        with open(f_path) as f_in:
            for line in f_in.read().splitlines():
                dna_arr, label = convert_data(line)
                data.append({
                    'x': dna_arr,
                    'y': label
                })

    return data


def dnn_model_fn(features, labels, mode):
    """
    Multi-layer DNN with dense hidden layers & relu activation.

    :param features: the features to train/test on
    :param labels: the labels on each DNA to check against prediction
    :param mode: can be train, 5fold, or test
    :returns type tf: tf goodness
    """

    # Shape input layer
    l0 = tf.reshape(features['x'], [-1, 40])

    # Define hidden layers
    l1 = tf.layers.dense(inputs=l0, units=40, activation=tf.nn.relu)
    l2 = tf.layers.dense(inputs=l1, units=40, activation=tf.nn.relu)
    l3 = tf.layers.dense(inputs=l2, units=40, activation=tf.nn.relu)
    l4 = tf.layers.dense(inputs=l3, units=40, activation=tf.nn.relu)

    # Define output (logit) layer
    logits = tf.layers.dense(inputs=l4, units=6)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=6)
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


def train_mode(data):
    # Init estimator
    model_dir = os.path.join('model')
    sticky_classifier = tf.estimator.Estimator(
        model_fn=dnn_model_fn,
        model_dir=model_dir
    )

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=1000
    )

    features = np.asarray([d['x'] for d in data])
    labels = np.asarray([d['y'] for d in data], dtype=np.float32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': features},
        y=labels,
        batch_size=MINIBATCH_SIZE,
        num_epochs=None,
        shuffle=True
    )
    sticky_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )

    # TODO Eval model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": features},
        y=labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = sticky_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    print('Processing complete!')
    print(f'Total items trained on: {len(data)}')


def k_fold_mode(args, k=5):
    data = load_data(args.data_folder)
    subset_size = int(len(data) / k)
    subsets = [data[i:i + subset_size] for i in range(0, len(data), subset_size)]

    # if size of data isn't divisible by 5, have a larger kth subset
    if len(data) % k != 0:
        subsets[k-1] = subsets[k-1] + subsets[k]
        del subsets[k]

    # perform cross validation
    for i in range(k):
        # create copy and then remove subset i
        training_set = list(subsets)
        del training_set[i]
        # train on training_set
        train_mode(training_set)

        # test on subsets[i]
        # train_mode(subsets[i])
        pass


def main(args):
    # call function based on mode
    if args.mode == 'train':
        # Load training data
        data = load_data(args.data_folder)
        train_mode(data)
    elif args.mode == '5fold':
        k_fold_mode(args)
    elif args.mode == 'train':
        pass


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

