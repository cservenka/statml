import os
import numpy as np

from sklearn import svm
from iii_2_1 import readFile, normalize

data = readFile(os.path.dirname(__file__) + '/../../data/parkinsonsTrainStatML.dt')
print data
print len(data[0])