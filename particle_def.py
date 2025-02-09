import sys

import numpy as np

polymerLenght = 50
headN = 1


def polymergenerator(polymerl, headl):
    polymer = np.array((range(polymerl), np.zeros(50), np.zeros(50))).T
    # heads = np.array([range(headl), np.zeros(headl), np.zeros(headl)]).T
    heads = np.array((10, 1, 1))
    return polymer, heads


def forcefield(a, b, epsilon=1, sigma=1):
    dif = a - b
    dist = np.sqrt(np.sum(np.square(dif), axis=1))
    f = 4 * epsilon * ((sigma / dist)**12 - (sigma / dist)**6)
    vector = dif / np.array((dist, dist, dist)).T
    return dist, dif, vector


polymerX, headX = polymergenerator(polymerLenght, headN)
distance, forces, vectors = forcefield(polymerX, headX)
np.set_printoptions(threshold=sys.maxsize)
print(distance, f"\n", forces, f"\n", vectors)
