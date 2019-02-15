import numpy as np
import csv
from math import log

lst = []
# Store all the rows into a list
with open('ml-bugs.csv') as csvfile:
    file = csv.reader(csvfile)
    i = 0
    for row in file:
        if (i != 0):
            lst.append((row[0], row[1], row[2]))
        i = 1

# Calculate overall entropy
def overall_entropy(lst):
    # Split the rows into two lists
    Mobugs = [] # a list which label is mobugs
    Lobugs = [] # a list which label is lobugs

    for item in lst:
        if item[0] == 'Mobug':
            Mobugs.append(item)
        else:
            Lobugs.append(item)
    mobugs_num = len(Mobugs)
    lobugs_num = len(Lobugs)

    p1 = mobugs_num / len(lst)
    p2 = lobugs_num / len(lst)
    # print('mobugs:', mobugs_num)
    # print('lobugs:', lobugs_num)
    return -p1 * log(p1, 2) - p2 * log(p2, 2)


def devide_by_color(color):
    # if splitting criteria is color,
    # calculate information gain.
    color = []
    not_color = []
    # Split into two branches, color and no color
    for item in lst:
        if (item[1] == color):
            color.append(item)
        else:
            not_color.append(item)
    print(len(color))
    print(len(not_color))
    mobugs_color = []
    lobugs_color = []

    mobugs_not_color = []
    lobugs_not_color = []

    for item in color:
        if item[0] == 'Mobug':
            mobugs_color.append(item)
        else:
            lobugs_color.append(item)

    for item in not_color:
        if item[0] == 'Mobug':
            mobugs_not_color.append(item)
        else:
            lobugs_not_color.append(item)
    print('mobugs_color', len(mobugs_color))
    print('lobugs_color', len(lobugs_color))
    print('mobugs_not_color', len(mobugs_not_color))
    print('mobugs_not_color', len(mobugs_not_color))

    if len(color) == 0:
        entropy_color = 0
    else:
        p_mobugs_color = len(mobugs_color) / len(color)
        p_lobugs_color = len(lobugs_color) / len(color)
        entropy_color = -p_mobugs_color * log(p_mobugs_color, 2) - p_lobugs_color * log(p_lobugs_color, 2)

    if (len(not_color) == 0):
        entropy_not_color = 0
    else:
        p_mobugs_not_color = len(mobugs_not_color) / len(not_color)
        p_lobugs_not_color = len(lobugs_not_color) / len(not_color)
        entropy_not_color = -p_mobugs_not_color * log(p_mobugs_not_color, 2) - p_lobugs_not_color * log(p_lobugs_not_color, 2)


    return entropy_color, entropy_not_color

def calculate_entropy(color):
    first_entropy = overall_entropy(lst)
    print(first_entropy)
    entropy_color, entropy_not_color = devide_by_color(color)
    return first_entropy - (entropy_color + entropy_not_color) / 2

print(calculate_entropy('Brown'))
