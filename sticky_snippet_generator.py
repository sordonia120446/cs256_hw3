"""
Gene Snippet Detection with Tensorflow

:authors Jason, Nick, Sam
"""

import argparse
import os
import random

def generate_strings(args, str_len=40):
    '''
    Generate any number of DNA strings and write to file.

    :param args: User command line args
    :param alphabet: The alphabet to use for each string
    :param sticks_to: A dictionary describing which letters stick to which
    :param str_len: The length of each string
    :param default_mutation: The default mutation rate applied to chars greater than from_ends from start/end of string
    '''

    if not 0 <= args.mutation_rate <= 1:
        raise Exception('mutation_rate must be a valid probability!')

    strings = []
    for i in range(args.num_snippets):
        if i == 0:
            # Create the base string & mutate it
            base_str = generate_palindrome(str_len)
            strings.append(mutate_string(base_str, args.mutation_rate, args.from_ends))
        else:
            # Create a mutated string based on the last string according to mutation_rate and from_ends
            strings.append(mutate_string(strings[i - 1], args.mutation_rate, args.from_ends))

        print(strings[i])

    if strings:
        # Write to file
        with open(args.output_file, 'w') as f:
            for str in strings:
                f.write(str + ',' + get_stickiness(str) + '\n') # Needs labels

def generate_palindrome(str_len, alphabet=['A', 'B', 'C', 'D'], sticks_to={'A': 'C', 'B': 'D', 'C': 'A', 'D': 'B'}):
    '''
    Generate a palindrome in the specified alphabet

    :param str_len: Length of the palindrome to generate
    :param alphabet: The alphabet to use (should be an even number)
    :param sticks_to: The relationships between letters
    :return:
    '''

    if not str_len % 2 == 0:
        str_len += 1

    palindrome = ''
    for i in range(str_len):
        if i < str_len // 2:
            palindrome += random.choice(alphabet)
        else:
            palindrome += sticks_to[palindrome[str_len // 2 - (i % (str_len // 2) + 1)]]

    return palindrome

def mutate_string(dna_str, mutation_rate, from_ends, alphabet=['A', 'B', 'C', 'D'], default_mutation=0.5):
    '''
    Mutate a given string based on mutation_rate and default_mutation

    :param dna_str: String to be mutated
    :param mutation_rate: Rate to be applied to characters from_ends from ends
    :param from_ends: Distance from ends to apply mutation_rate to
    :param alphabet: The alphabet to use
    :param default_mutation: The mutation rate to use beyond from_ends
    :return: The mutated string
    '''

    mutated_str = ''

    for i, c in enumerate(dna_str):
        choices = alphabet.copy()
        choices.remove(c)

        if i < from_ends or i >= len(dna_str) - from_ends:
            mutation = mutation_rate
        else:
            mutation = default_mutation

        if random.random() <= mutation:
            mutated_str += random.choice(choices)
        else:
            mutated_str += c

    return mutated_str


def get_stickiness(dna_str, sticks_to={'A': 'C', 'B': 'D', 'C': 'A', 'D': 'B'}):
    '''
    Determines the stickiness of a string and returns its label

    :param dna_str: The string to process
    :param sticks_to: A dictionary describing which letters stick to which
    :return: A label indicating the string's stickiness
    '''

    stick = 0
    for i in range(len(dna_str) // 2):
        try:
            if sticks_to[dna_str[i]] == dna_str[len(dna_str) - 1 - i]:
                stick = i + 1
            else:
                break
        except KeyError:
            raise Exception('Invalid string')

    if stick == len(dna_str) // 2:
        return 'STICK_PALINDROME'
    elif stick == 0:
        return 'NONSTICK'
    elif stick == 1 or stick == 2:
        return '12-STICKY'
    elif stick == 3 or stick == 4:
        return '34-STICKY'
    elif stick == 5 or stick == 6:
        return '56-STICKY'
    elif stick >= 7:
        return '78-STICKY'
    else:
        raise Exception('Something bad happened')


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
