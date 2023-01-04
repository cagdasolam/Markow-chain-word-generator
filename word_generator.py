import random

import numpy as np
import matplotlib.pyplot as plt


def read_file(filename):
    """
    Read lines from a file and store them in a list
    :param filename: str
    :return: str[]
    """
    words = []
    # Open the file in read-only mode
    with open(filename, "r") as file:
        # Read all lines in the file
        lines = file.readlines()

        # Loop through each line in the file
        for line in lines:
            words.append(line.rstrip() + '*')

    return words


def l0(word_list):
    """
    calculates the probability of a letter being the first letter by looking at the words in a given list
    :param word_list: list of string
    :return: a dictionary letter as key probability as value
    """
    # Create a dictionary to store the probabilities of each letter
    letter_probabilities = {}

    # Loop through each line in the file
    for word in word_list:
        # Get the first letter of the word
        first_letter = word[0]

        # Increment the count for this letter in the dictionary
        if first_letter in letter_probabilities:
            letter_probabilities[first_letter] += 1
        else:
            letter_probabilities[first_letter] = 1

    # Loop through each letter in the dictionary
    for letter, count in letter_probabilities.items():
        # Compute the probability of this letter by dividing its count by the total number of words in the file
        probability = count / len(word_list)

        # Set the probability of this letter in the dictionary
        letter_probabilities[letter] = probability

    letter_probabilities['*'] = 0

    # Return the dictionary of letter probabilities~~~
    return letter_probabilities


def create_matrix(word_list):
    """
    By looking at the words in a given list,
    it calculates the probability that another letter will appear after a letter and saves the result in a matrix
    :param word_list: list of string
    :return: probability matrix
    """
    # Initialize a 27 by 27 matrix of zeros
    matrix = np.zeros((27, 27))

    for word in word_list:
        for i in range(len(word) - 1):
            # Get the current and next characters
            current_char = word[i]
            next_char = word[i + 1]

            # If the current character is a letter, increment the count for the
            # corresponding next character in the matrix
            if current_char.isalpha():
                # If the next character is a letter, increment the count
                # in the corresponding column in the matrix
                if next_char.isalpha():
                    matrix[ord(current_char) - 97, ord(next_char) - 97] += 1
                # If the next character is the end of a word (new line),
                # increment the count in the last column of the matrix
                elif next_char == '*':
                    matrix[ord(current_char) - 97, 26] += 1

    # Divide each element in the matrix by the total number of characters that follow the letter
    # represented by the corresponding row to calculate the probability that a given character follows
    # the letter represented by the row
    # print(matrix)
    for i in range(26):
        total = 0
        for j in range(len(matrix[i])):
            total += matrix[i][j]
        for j in range(len(matrix[i])):
            matrix[i][j] /= total

    return matrix


def average_length_word(word_list):
    """
    calculate average length of words in given list
    :param word_list: string list
    :return: float
    """
    total_length = 0
    total_word = 0

    for word in word_list:
        total_length += len(word.rstrip('*'))
        total_word += 1

    total_length /= total_word

    return total_length


def calc_prior_prob1(words, n):
    """
    calculate the probability of a letter nth place for every word in list
    :param words: string array
    :param n: int order of letters
    :return: dictionary sorted by alphabetic order
    """
    # Extract the Nth letters of the words in the word list
    letters = []
    for word in words:
        if n < len(word):
            if word[n] == '*':
                continue
            letters.append(word[n])

    # Calculate the frequencies of the letters
    letter_counts = {}
    for letter in set(letters):
        letter_counts[letter] = letters.count(letter)

    # Calculate the percentage distribution of the letters
    letter_distribution = {}
    for letter, count in letter_counts.items():
        letter_distribution[letter] = count / len(letters)

    letter_distribution['*'] = 0

    return dict(sorted(letter_distribution.items()))


def plot_calc_prior_prob1(words, n):
    """
    plot function for calc_prior_prob1
    :param words: string array
    :param n: int order of letters
    :return:
    """
    for i in range(1, n + 1):
        sorted_dict = calc_prior_prob1(words=words, n=i)
        # Create the bar graph
        plt.title('calcpriorprob1 for N = {}'.format(i))
        plt.bar(sorted_dict.keys(), sorted_dict.values())
        plt.show()


def calc_prior_prob2(p0_dict, matrix, n):
    """
    calculate the probability of a letter nth place for every word in list according to the letter that comes before it
    :param p0_dict: dictionary that stores probabilities of a letter being a 'first letter'
    :param matrix: a matrix that stores probabilities that another letter will appear after a letter
    :param n: int order of letters
    :return: dictionary sorted by alphabetic order
    """
    # shallow copy of p0_dict
    letter_probs = p0_dict.copy()
    # convert the p0 dict values to an array
    p0_array = []

    for value in letter_probs.values():
        p0_array.append(value)

    # dot product two array
    for i in range(n):
        temp = np.dot(p0_array, matrix)
        p0_array = temp

    i = 0
    for key in letter_probs:
        letter_probs[key] = p0_array[i]
        i += 1

    # sort the letter distribution for graph
    return dict(sorted(letter_probs.items()))


def plot_calc_prior_prob2(p0_dict, matrix, n):
    """
    plot function for calc_prior_prob2
    :param p0_dict: dictionary that stores probabilities of a letter being a 'first letter'
    :param matrix: a matrix that stores probabilities that another letter will appear after a letter
    :param n: int order of letters
    :return:
    """
    for i in range(1, n + 1):
        sorted_dict = calc_prior_prob2(p0_dict, matrix, i)
        # Create the bar graph
        plt.title('calcpriorprob2 for N = {}'.format(i))
        plt.bar(sorted_dict.keys(), sorted_dict.values())
        plt.show()


def calc_word_prob(p0_dict, matrix, word):
    """
    Calculates the probability of a given word occurring in a row for each letter that makes up that word
    :param p0_dict: dictionary that stores probabilities of a letter being a 'first letter'
    :param matrix: a matrix that stores probabilities that another letter will appear after a letter
    :param word: string
    :return: string
    """
    # shallow copy of p0_dict
    letter_probs = p0_dict.copy()

    res = letter_probs[word[0]]
    for i in range(1, len(word)):
        letter = word[i]
        previous_letter = word[i - 1]
        if letter == '*':
            res *= matrix[ord(previous_letter) - 97][26]
        else:
            res *= matrix[ord(previous_letter) - 97][ord(letter) - 97]

    return 'probability of the given word \'{}\' is \'{}\''.format(word, res)


def generate_words(p0_dict, matrix, m):
    """
    choosing letters according to given probability matrices
    Generates as many words as the given parameter m
    :param p0_dict: dictionary that stores probabilities of a letter being a 'first letter'
    :param matrix: a matrix that stores probabilities that another letter will appear after a letter
    :param m: number of words that will be generated
    :return: list of words
    """
    # shallow copy of p0_dict
    letter_probs = p0_dict.copy()
    # set alphabet
    alphabet = list(letter_probs.keys())
    words = []

    for i in range(m):
        letter_array = []
        first_letter = random.choices(alphabet, weights=letter_probs.values(), k=1)[0]  # choose first letter
        letter_array.append(first_letter)
        letter = first_letter
        n = 1
        # until the end of word character be chosen, choose a letter and append it list
        while letter != '*':
            weight = matrix[ord(letter) - 97]
            letter = random.choices(alphabet, weights=weight, k=1)[0]
            letter_array.append(letter)
            n += 1

        # generate word string
        word = ''
        for letter in letter_array:
            word += letter

        words.append(word)

    return words


def bonus_create_matrix_pair(words):
    """
    for bonus part
    By looking at the words in a given list,
    it calculates the probability that another letter will appear after a letters of pair and saves the result in a matrix
    :param words: String array
    :return: probability matrix
    """
    probabilities = np.zeros((27 * 27, 27))

    # Create dictionaries that hold numbers of binaries followed by letters
    pair_counts = {}
    letter_after_pair_counts = {}
    for word in words:
        for i in range(len(word) - 1):
            pair = word[i:i+2]
            if pair not in pair_counts:
                pair_counts[pair] = 0
                letter_after_pair_counts[pair] = {}
            pair_counts[pair] += 1

            # Count the letter that comes after the binary
            if i < len(word) - 2:
                letter_after = word[i + 2]
            else:
                letter_after = '*'

            if letter_after not in letter_after_pair_counts[pair]:
                letter_after_pair_counts[pair][letter_after] = 0
            letter_after_pair_counts[pair][letter_after] += 1

    # Calculate the probability of being found for each binary and assign it to the matrix
    pair_index = 0
    for pair, count in pair_counts.items():
        letter_after_counts = letter_after_pair_counts[pair]
        for letter_after, after_count in letter_after_counts.items():
            # probability of finding the letter following the pair
            probability = after_count / count
            # Assign to trailing column for word break
            if letter_after == '*':
                probabilities[pair_index][-1] = probability
            else:
                # Find the index of the letter for the other letters and assign the probability there
                letter_index = ord(letter_after) - ord('a')
                probabilities[pair_index][letter_index] = probability

        pair_index += 1

    return probabilities


def generate_random_word(p0_dict, matrix, matrix2, m):
    """
    choosing letters according to given probability matrices
    Generates as many words as the given parameter m
    :param p0_dict: dictionary that stores probabilities of a letter being a 'first letter'
    :param matrix: a matrix that stores probabilities that another letter will appear after a letter
    :param matrix2: a matrix that stores probabilities that another letter will appear after a pair
    :param m: number of words that will be generated
    :return: list of words
    """
    # shallow copy of p0_dict
    letter_probs = p0_dict.copy()
    alphabet = list(letter_probs.keys())
    words = []

    for i in range(m):
        letter_array = []
        first_letter = random.choices(alphabet, weights=letter_probs.values(), k=1)[0]
        second_letter = random.choices(alphabet, weights=matrix[ord(first_letter) - 97])[0]
        if second_letter == '*':
            letter_array.append(first_letter)
            letter_array.append(second_letter)
            word = ''
            for letter in letter_array:
                word += letter

            words.append(word)
            continue

        letter_array.append(first_letter)
        letter_array.append(second_letter)
        letter = first_letter
        n = 1
        while second_letter != '*':
            weight = matrix2[ord(first_letter) - 97 + ord(second_letter) - 97]
            letter = random.choices(alphabet, weights=weight, k=1)[0]
            letter_array.append(letter)
            first_letter = second_letter
            second_letter = letter
            n += 1

        word = ''
        for letter in letter_array:
            word += letter

        words.append(word)

    return words


def main():
    words = read_file('corncob_lowercase.txt')

    l0_dict = l0(word_list=words)
    matrix = create_matrix(word_list=words)
    average_length = average_length_word(word_list=words)

    print('P(L0):\n', l0_dict)
    print()
    print('-----matrix-------')
    print(matrix)
    print()
    print('average length for the given text is: ', average_length)
    print()

    plot_calc_prior_prob1(words=words, n=5)
    plot_calc_prior_prob2(l0_dict, matrix, 5)

    print(calc_word_prob(p0_dict=l0_dict, matrix=matrix, word='sad*'))
    print(calc_word_prob(p0_dict=l0_dict, matrix=matrix, word='exchange*'))
    print(calc_word_prob(p0_dict=l0_dict, matrix=matrix, word='antidisestablishmentarianism*'))
    print(calc_word_prob(p0_dict=l0_dict, matrix=matrix, word='qwerty*'))
    print(calc_word_prob(p0_dict=l0_dict, matrix=matrix, word='zzzz*'))
    print(calc_word_prob(p0_dict=l0_dict, matrix=matrix, word='ae*'))

    print('\ngenerated words for first part:')
    print(generate_words(p0_dict=l0_dict, matrix=matrix, m=10))

    synthetic_data_set = generate_words(p0_dict=l0_dict, matrix=matrix, m=100000)
    average = average_length_word(synthetic_data_set)
    print('\naverage length of the generated synthetic dataset of size 100000 ', average)

    matrix2 = bonus_create_matrix_pair(words)

    print('\ngenerated words for bonus part:')
    print(generate_random_word(l0_dict, matrix, matrix2, 10))


if __name__ == '__main__':
    main()
