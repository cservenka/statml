from __future__ import division

import os
import numpy as np

from i_3_1 import *
from i_3_2 import *

def normalize(lst, targetMax, targetMin):
    currentMax = max(lst)
    currentMin = min(lst)

    a = (targetMax - targetMin) / (currentMax - currentMin)
    b = targetMax - a * currentMax

    return map(lambda value: a * value + b, lst)

def transpose(data):
    return np.array(data).T.tolist()

def normalizeDataset(data):
    npData = np.array(data).T
    normalizedData = []
    normalizedData.append(normalize(npData[0].tolist(), 0.5, -0.5))
    normalizedData.append(normalize(npData[1].tolist(), 0.5, -0.5))
    normalizedData.append(npData[2].tolist())

    return transpose(normalizedData)

if __name__ == '__main__':
    train = readFile(os.path.dirname(__file__) + '/../../data/IrisTrain2014.dt')
    test = readFile(os.path.dirname(__file__) + '/../../data/IrisTest2014.dt')

    normalized = transpose(normalizeDataset(train))
    print np.var(normalized[1])