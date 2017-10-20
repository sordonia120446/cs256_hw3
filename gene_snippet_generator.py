"""
Gene Snippet Detection with Tensorflow

:authors Jason, Nick, Sam
"""

import argparse
import random

import tensorflow as tf

# TODO Generate strings

def generate_strings(args, alphabet=['A', 'B', 'C', 'D'], str_len=40, default_mutation=0.5):
    '''
    Generate any number of DNA strings and write to file.

    :param args: User command line args
    :param alphabet: The alphabet to use for each string
    :param str_len: The length of each string
    :param default_mutation: The default mutation rate applied to chars greater than from_ends from start/end of string
    '''

    if not 0 <= args.mutation_rate <= 1:
        raise Exception('mutation_rate must be a valid probability!')

    strings = []
    for i in range(args.num_snippets):
        if i == 0:
            # Create the base string
            strings.append(''.join(random.choice(alphabet) for _ in range(str_len)))
        else:
            # Create a mutated string based on the last string according to mutation_rate and from_ends
            strings.append(
                ''.join(
                    random.choice(alphabet)
                    if random.random() <= (args.mutation_rate if j < args.from_ends or j >= len(strings[i - 1]) - args.from_ends else default_mutation)
                    else c
                    for j, c in enumerate(strings[i - 1])
                )
            )

        print(strings[i])

    if strings:
        # Write to file
        with open(args.output_file, 'w') as f:
            for str in strings:
                f.write(str + '\n') # Needs labels


# TODO Label data w/ stickiness

def get_stickiness(dna_str):
    '''
    Determines the stickiness of a string and returns its label

    :param dna_str: The string to process
    :return: A label indicating the string's stickiness
    '''

    raise NotImplementedError

"""CLARGS"""
parser = argparse.ArgumentParser(
    description='CS 256 Homework 3',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

# Add CLARGS
parser.add_argument(
    'num_snippets',
    type=int,
    action='store',
    help='Number of number of gene snippets to generate.'
)
parser.add_argument(
    'mutation_rate',
    type=float,
    action='store',
    help='Float between 0 and 1 representing the odds that '
        'a character gets mutated to a random other character.'
)
parser.add_argument(
    'from_ends',
    type=int,
    action='store',
    help='Distance from either the start or end '
        'of the string to apply the mutation rate to.'
)

parser.add_argument(
    'output_file',
    action='store',
    help='Name of output file'
)


if __name__ == '__main__':
    args = parser.parse_args()
    generate_strings(args)