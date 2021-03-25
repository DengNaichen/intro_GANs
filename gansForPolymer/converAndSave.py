import pandas as pd
import re
import matplotlib.pyplot as plt


def str_to_int(item):
    c = []
    j = 0
    for i in range(3):
        if i < 2:
            if item[j] == '-':
                a = int(item[j] + item[j + 1])
                j += 2
            else:
                a = int(item[j])
                j += 1
        elif i == 2:
            a = int(item[j:])
        c.append(a)
    return c


def convert2dirction(a, b):
    if a[0] - b[0] == 1:
        # left
        return -2
    elif a[0] - b[0] == -1:
        # right
        return -1
    elif a[1] - b[1] == -1:
        # up
        return 1
    else:
        # down
        return 2


def plot_polymer(coordinate):
    for i in range(len(coordinate) - 1):
        plt.plot([coordinate[i][0], coordinate[i + 1][0]], [coordinate[i][1], coordinate[i + 1][1]])
    fig = plt.gcf()
    plt.axis('off')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    fig.savefig("test.png")
    plt.show()


def read_dat_file(filename):
    coordinate = []
    strinfo = re.compile(' ')
    for line in open(filename, "r"):
        item = line.rstrip()
        if len(item) == 0:
            continue
        item = strinfo.sub('', item)
        str_to_int(item)
        # convert str to int list
        item = str_to_int(item)
        coordinate.append(item)
    return coordinate


def arrange_direction(coordinate, num):
    direction_all = []
    for i in range(num):
        direction = []
        for j in range(15):
            a = convert2dirction(coordinate[j + (i + 15)], coordinate[(j + 1) + (i * 15)])
            direction.append(a)
        direction_all.append(direction)
    return direction_all


def save_file(direction_all):
    df = pd.DataFrame(direction_all)
    df.to_csv("coordinate.csv", index=True, header=True)


coordinate = read_dat_file('Coordinates.dat')
direction_all = arrange_direction(coordinate, 100000)
save_file(direction_all)

