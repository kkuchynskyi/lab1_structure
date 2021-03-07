import math
import json
import os
from collections import OrderedDict, defaultdict
from numba import jit

import cv2
import numpy as np
from PIL import Image

# +
def create_prob_dict(alphabet_list, path='./lab1_data/frequencies.json'):
    """
    Open json-file and create corresponding probability dictionary
    :param alphabet: a list of string symbols
    :param path: path to json
    :return: a dictionary, where keys: all possible bigrams,
        values: probabilites of corresponding bigrams
    """
    with open(path) as f:
        prob_dict = json.load(f)
    total_count = 0
    for v in prob_dict.values():
        total_count += v
    for k in prob_dict.keys():
        prob_dict[k] /= total_count

    for i in alphabet_list:
        for j in alphabet_list:
            if i + j not in prob_dict.keys():
                prob_dict[i + j] = 1e-128
    dict_prob_cond = defaultdict(lambda x: 0)
    for k, v in prob_dict.keys():
        if len(k) == 2:
            dict_prob_cond[k[0]] += v

    for k, v in prob_dict.keys():
        if len(k) == 2:
            prob_dict[k] /= dict_prob_cond[k[0]]
    return prob_dict


def create_char_to_arr_dict(alphabet_list, w, h, path='./lab1_data/alphabet/'):
    """
    Create array, which contains a matrix representation of chars
    :param alphabet_list: a list of string symbols
    :param path: path to images
    :return: a dictionary, where keys: a char from 'alphabet_list',
        values: a corresponding matrix
    """
    char_to_arr_dict = dict()
    for character_name in alphabet_list:
        if character_name == " ":
            path_to_image = os.path.join(path, "space.png")
        else:
            path_to_image = os.path.join(path, character_name + ".png")
        character_arr = np.array(Image.open(path_to_image), dtype=np.uint8)
        character_arr = cv2.resize(character_arr, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        char_to_arr_dict[character_name] = character_arr.astype(np.bool)
    return char_to_arr_dict


def read_input_sentence(index, path='./lab1_data/input/'):
    """
    Read an input image as numpy array
    :param index: int in range [0, 1, 2, ..., 15]
    :param path: path to input images
    :return: numpy array
    """
    list_input_txt = os.listdir(path)
    for i, name in enumerate(list_input_txt):
        print(i, name)
    input_txt_file = list_input_txt[index]
    path_to_image = "./lab1_data/input/{}".format(input_txt_file)
    input_arr = np.array(Image.open(path_to_image), dtype=np.bool)
    print("You have read = ", input_txt_file)
    return input_arr

# +
@jit(nopython=True)
def calculate_ln_prob(noised_arr, label_arr, p):
    """
    Calculate probability
    :param noised_arr: n*m numpy matrix with 0 and 1
    :param label_arr: n*m numpy matrix with 0 and 1
    :param p: float probability in (0, 1)
    :return: float
    >>> calculate_ln_prob(np.array([[1]]), np.array([[0]]), 0.5)
    -0.6931471805599453
    >>> calculate_ln_prob(np.array([[1]]), np.array([[0]]), 0.2)
    -1.6094379124341003
    >>> calculate_ln_prob(np.array([[1, 0]]), np.array([[0, 1]]), 0.75)
    0.810930216216329
    >>> calculate_ln_prob(np.array([[1, 1]]), np.array([[1, 1]]), 0.75)
    -1.3862943611198906
    >>> calculate_ln_prob(np.array([[0, 0]]), np.array([[0, 0]]), 0.75)
    -1.3862943611198906
    >>> calculate_ln_prob(np.array([[1, 0],[ 1, 0]]), np.array([[0, 1],[1, 0]]), 0.75)
    0.810930216216329
    >>> calculate_ln_prob(np.array([[1, 0],[ 1, 0]]), np.array([[1, 1],[1, 1]]), 0.75)
    0.810930216216329
    >>> calculate_ln_prob(np.array([[1, 1],[ 1, 1]]), np.array([[1, 1],[1, 1]]), 0.75)
    -1.3862943611198906
    """
    assert noised_arr.shape == label_arr.shape
    assert 0 < p < 1
    n, m = noised_arr.shape
    xor_sum = (noised_arr ^ label_arr).sum()
    return math.log(p/(1-p))*xor_sum + math.log(1-p)*n*m


#########################################################################################

def bernoulli_noise(true_image, p):
    assert len(true_image.shape) == 2
    assert 0 < p < 1
    noise_matrix = np.zeros(true_image.shape, np.uint8)
    for i in range(true_image.shape[0]):
        for j in range(true_image.shape[1]):
            if np.random.rand() <= p:
                noise_matrix[i, j] = 1
    return true_image ^ noise_matrix


def create_noised_array(str_to_convert, char_to_arr_dict, h, w, p):
    true_arr = np.zeros((h, w * len(str_to_convert)), np.uint8)
    for i in range(len(str_to_convert)):
        char = str_to_convert[i].lower()
        char_arr = char_to_arr_dict[char].astype(np.uint8)
        true_arr[:, i * w:(i + 1) * w] = char_arr
        noised_input_arr = bernoulli_noise(true_arr, p)
    return noised_input_arr


def calculate_F_next(F_prev, step, char_to_arr_dict, input_arr, prob_dict, p, n):
    """
    Create the data structure (dictionary of dictionaries), which being used
    for computation. Keys: integers from 0, to m+1. Values: a dictionary
    where keys: a char (exist 'word', where the last symbol is this char),
    where values:  [(float) a probability of the 'word',
    (string)'word', string where the last symbol is a key of a dict
    (int)step on which the word has been created]
    :param char_to_arr_dict: a dictionary, where keys: a char from 'alphabet_list',
        values: a corresponding matrix
    :param input_arr: n*m numpy array
    :param char_distr: a dictionary, where keys: all possible bigrams,
        values: probabilites of corresponding bigrams
    :param p: float probability in (0, 1)

    """
    # calculate F0_1
    for k_next, k_next_arr in char_to_arr_dict.items():
        if k_next != " ":
            current_width = k_next_arr.shape[1]
            input_character = input_arr[:, step*current_width:(step+1)*current_width]
            prob = calculate_ln_prob(input_character, k_next_arr, p) #+ np.log(prob_dict[prev_k+k_next])
            F_prev += prob
    return F_prev

def calculate_F(i_start, i_end, char_to_arr_dict, input_arr, prob_dict, p):
    if i_start == 0:
        F = 0
    else:
        space_char = input_arr[:, (i_start-1)*30:i_start*30]
        F = (len(char_to_arr_dict.keys()))*calculate_ln_prob(space_char, char_to_arr_dict[" "], p)*2

    for k, k_arr in char_to_arr_dict.items():
        if k != " ":
            current_width = k_arr.shape[1]
            input_character = input_arr[:, i_start*current_width:(i_start+1)*current_width]
            prob = calculate_ln_prob(input_character, k_arr, p) + np.log(prob_dict[" "+k])
            F += prob
    n = 1
    for i_intermidiate in range(i_start+1, i_end+1, 1):
        n *= (len(char_to_arr_dict.keys()) - 1)
        F = calculate_F_next(F, i_intermidiate, char_to_arr_dict, input_arr, prob_dict, p, n)
    return F


#@jit(nopython=True)
def calculate_dynamic(n, char_to_arr_dict, noised_input_arr, prob_dict, p):
    F_arr = np.zeros((n,), dtype=np.float64)
    F_previous = np.zeros((n,), dtype=np.float64)
    max_index = None
    for i in range(1, n, 1):
        max_value = -np.inf
        # loop through in range [1, i-1]
        for j in range(1, i-1):
            F = F_arr[j - 1] + calculate_F(j + 1, i, char_to_arr_dict, noised_input_arr, prob_dict, p)
            if F > max_value:
                max_value = F
                max_index = j
        F_arr[i] = max_value
        F_previous[i] = max_index
    return F_previous


def restore(F, input_str):
    last_char = F[-1]
    spaces = list()
    spaces.append(last_char)
    for i in range(len(F)):
        last_char = F[int(last_char)]
        if last_char == 1:
            break
        spaces.append(last_char)
    reversed(spaces)
    edited_list = list(input_str)
    for i in spaces:
        edited_list[int(i)] = "!"
    edited_str = ''.join(edited_list)
    return edited_str

