import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from functools import reduce



def clear_left(sorted_distribution, vals_part, width_part, cut_coeff):
    
    while True:
        vals_count = int(len(sorted_distribution) * vals_part)

        distribution_width = sorted_distribution[-1] - sorted_distribution[0]
        shorted_part_width = sorted_distribution[vals_count] - sorted_distribution[0]
        shorted_part = shorted_part_width / distribution_width

        if shorted_part < width_part: 
            return sorted_distribution

        trash_sep_width = shorted_part_width / vals_count * cut_coeff

        for i in range(vals_count):
            left_width_diff = sorted_distribution[vals_count - i] - sorted_distribution[vals_count - i - 1]

            if abs(left_width_diff) > abs(trash_sep_width):
                sorted_distribution = sorted_distribution[vals_count - i :]
                break

    return sorted_distribution


def clear(sorted_distribution, vals_part, width_part, cut_coeff):
    sorted_distribution = clear_left(sorted_distribution, vals_part, width_part, cut_coeff)
    sorted_distribution = clear_left(sorted_distribution[::-1], vals_part, width_part, cut_coeff)[::-1]

    return sorted_distribution


def get_x_axis(sorted_distribution, step):
    x_start = sorted_distribution[0] // step * step
    if x_start >= sorted_distribution[0]:
        x_start -= step

    current = x_start
    result = []

    while current <= sorted_distribution[-1]:
        result.append(current)
        current += step

    return result


def get_heights(x_axis, sorted_distribution):
    heights = [0] * len(x_axis)
    current_index = len(x_axis) - 1

    for value in reversed(sorted_distribution):
        if value > x_axis[current_index]:
            heights[current_index] += 1
        else:
            if value == x_axis[current_index]:
                heights[current_index] += 0.5
                heights[current_index - 1] += 0.5
            current_index -= 1
    
    return heights


def get_expected_value(distribution):
    return reduce(lambda x, y: x + y, distribution) / len(distribution)


def get_variance(distribution, expected_value):
    return reduce(lambda x, y: x + y * y, distribution, 0) / len(distribution) - expected_value ** 2


def lower_count(sorted_distribution, value):
    if len(sorted_distribution) == 0:
        return 0

    median = len(sorted_distribution) // 2

    if sorted_distribution[median] < value:
        return median + 1 + lower_count(sorted_distribution[median + 1 :], value)
    else:
        return lower_count(sorted_distribution[: median], value)


def distribution_function(sorted_distribution, value):
    return lower_count(sorted_distribution, value) / len(sorted_distribution)


def read_csv(filepath):
    with open(filepath) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        result = []
        for row in reader:
            result += row
    return result


def get_geometric_p_by_expected_value(expected_value):
    return 1 / expected_value


def get_geometric_p_by_variance(variance):
    return (-1 + (1 + 4 * variance) ** 0.5) / 2 / variance


def get_binomial_p_by_expected_value(expected_value, n):
    return expected_value / n


def get_binomial_p_by_variance(variance, n, switch = False):
    p = (n - (n ** 2 - 4 * n * variance) ** 0.5) / 2 / n

    if switch:
        return 1 - p
    else:
        return p


def get_uniform_low_high(expected_value, variance):
    high = (3 * variance) ** 0.5 + expected_value
    low = 2 * expected_value - high
    return low, high


def draw_hist(cleared_sorted_distribution, step):
    x_axis = get_x_axis(cleared_sorted_distribution, step)
    heights = get_heights(x_axis, cleared_sorted_distribution)

    plt.bar([value + step / 2 for value in x_axis], heights, step)


def draw_distribution_plot(sorted_distribution, density):
    left_border = math.floor(sorted_distribution[0])
    right_border = math.ceil(sorted_distribution[-1])

    xs = np.linspace(left_border, right_border, density)
    ys = [distribution_function(sorted_distribution, x) for x in xs]

    plt.plot(xs, ys)