import os
import numpy as np

from sklearn import svm

def readFile(filename):
	lines = [line.strip().split(' ') for line in open(filename)]
	lines_rotated = np.rot90(lines, 3)
	return (np.rot90([map(float, line) for line in lines_rotated[:-1]]), map(int, lines_rotated[-1])[::-1])

(data, result) = readFile(os.path.dirname(__file__) + '/../../data/parkinsonsTrainStatML.dt')
print data
print result