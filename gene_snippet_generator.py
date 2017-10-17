"""
Gene Snippet Detection with Tensorflow

:authors Jason, Nick, Sam
"""

import argparse

import tensorflow as tf


"""CLARGS"""
parser = argparse.ArgumentParser(
    description='CS 256 Homework 3',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

# Add CLARGS
parser.add_argument(
    'num_snippets',
    action='store',
    help='Number of number of gene snippets to generate.'
)
parser.add_argument(
    'mutation_rate',
    action='store',
    help='Float between 0 and 1 representing the odds that '
        'a character gets mutated to a random other character.'
)
parser.add_argument(
    'from_ends',
    action='store',
    help='Distance from either the start or end '
        'of the string to apply the mutation rate to.'
)


if __name__ == '__main__':
    args = parser.parse_args()

