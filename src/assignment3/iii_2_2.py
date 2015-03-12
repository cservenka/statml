import os
import itertools

import numpy as np

from sklearn import svm

def readFile(filename):
	lines = [line.strip().split(' ') for line in open(filename)]
	lines_rotated = np.rot90(lines, 3)
	return (np.rot90([map(float, line) for line in lines_rotated[:-1]]), map(int, lines_rotated[-1])[::-1])

(train_data, train_result) = readFile(os.path.dirname(__file__) + '/../../data/parkinsonsTrainStatML.dt')

C_range = [0.1, 1, 10, 100, 1000]
gamma_range = [0.001, 0.01, 0.1, 1, 10]
hyperparameter_grid = [pair for pair in itertools.product(C_range, gamma_range)]

#classifier = svm.SVC(C = 1.0, kernel = 'rbf')
#classifier.gamma = 1.0
#classifier.fit(data, result)