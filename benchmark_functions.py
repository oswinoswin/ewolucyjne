from math import cos, pi


def rastrigin(values):
    d = len(values)
    return 10 * d + sum([x * x - 10 * cos(2 * pi * x) for x in values]),
