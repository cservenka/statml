from __future__ import division

import numpy as np
import matplotlib.pyplot as mpl
import math

mu = -1
sigma = 1

def gauss(x):
    return (1 / (2 * math.pi * (sigma**2))**(1/2)) * np.exp(-1 / (2 * (sigma**2)) * ((x - mu)**2))

x = np.arange(-10.0, 10.0, 0.1)

mpl.plot(x, gauss(x))
mpl.ylabel('some numbers')

mpl.plot(x, gauss(x))

mu = 0
sigma = 2

mpl.plot(x, gauss(x))

mu = 2
sigma = 3

mpl.plot(x, gauss(x))

mpl.ylabel('gaussian distribution')
mpl.show()

if __name__ == '__main__':
	mpl.show()
