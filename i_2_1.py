from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import math

mu = -1
sigma = 1

def gauss(x):
    return (1 / (2 * math.pi * (sigma**2))**(1/2)) * np.exp(-1 / (2 * (sigma**2)) * ((x - mu)**2))

x = np.arange(-10.0, 10.0, 0.1)
y = np.arange(0.0, 10.0, 1.0)

plt.plot(x, gauss(x))
plt.ylabel('some numbers')
plt.show()
