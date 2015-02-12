import os.path
import math
import numpy as np
import operator
import collections
import copy

def readFile(filename):
	#Interpret first two columns as floats, last column as int
	toNum = lambda (a, b) : int(b) if a == 2 else float(b)

	#List of lists, each containing 2 floats and 1 int
	return [map(toNum, enumerate(line.strip().split(' '))) for line in open(filename)]

#Get euclidean distance between two data points
def euclideanMetric(a, b, length):
	#Length is in mm
	x1 = (a[0] - b[0]) ** 2
	#Width is in cm (so mult by 10)
	x2 = ((a[1] * 10) - (b[1] * 10)) ** 2
	return math.sqrt(x1 + x2) 

def getNeighbors(train, new, k):
	#Get first k elements of data set sorted ascendingly by distance from new data point
	sortPred = lambda e: euclideanMetric(new, e, len(new)-1)
	return sorted(train, key = sortPred)[:k]

#Get most common type amongst neighbors (as integer)
def getMajority(neighbors):
	return collections.Counter(map(lambda nb : nb[-1], neighbors)).most_common(1)[0][0]

#Get list of species (as int) from a data set
def getSpecies(test):
	return [e[-1] for e in test]

#Get 1 minus the mean of the 0-1 loss for each data entry
def getAccuracy(actual, predicted):
	loss = lambda (p, r): 0 if p is r else 1
	return 1 - np.mean(map(loss, zip(actual, predicted)))

if __name__ == '__main__':
	train = readFile(os.path.dirname(__file__) + '/../../data/IrisTrain2014.dt')
	test = readFile(os.path.dirname(__file__) + '/../../data/IrisTest2014.dt')

	for i in [1, 3, 5]:
		print "Training set K = %d Accuracy = %s" % (i, getAccuracy(getSpecies(train), [getMajority(getNeighbors(train, e, i)) for e in train]))
	print ""
	for i in [1, 3, 5]:
		print "Test set K = %d Accuracy = %s" % (i, getAccuracy(getSpecies(test), [getMajority(getNeighbors(train, e, i)) for e in test]))