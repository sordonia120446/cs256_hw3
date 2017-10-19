import argparse

import tensorflow as tf



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