import os.path
import math
import numpy as np
import operator
import collections

def readFile(filename):
	toNum = lambda (a, b) : int(b) if a == 2 else float(b)
	return [map(toNum, enumerate(line.strip().split(' '))) for line in open(filename)]

def euclideanMetric(a, b, length):
	distance = 0
	for i in xrange(length):
		distance += (a[i] - b[i]) ** 2
	return math.sqrt(distance)

def getNeighbors(train, new, k):
	train.sort(key = lambda e: euclideanMetric(new, e, len(new)-1))
	return [train[i] for i in xrange(k)]

def getMajority(neighbors):
	return collections.Counter(map(lambda nb : nb[-1], neighbors)).most_common(1)[0][0]

def getAccuracy(test, results):
	correct = []
	for (t, r) in zip(test, results):
		if t[-1] is r:
			correct.append(1)
		else:
			correct.append(0)
	return 1 - np.mean(correct)

if __name__ == '__main__':
	k = 5
	results = []
	train = readFile(os.path.dirname(__file__) + '/../../data/IrisTrain2014.dt')
	test = readFile(os.path.dirname(__file__) + '/../../data/IrisTest2014.dt')

	print getAccuracy(train, [getMajority(getNeighbors(train, e, k)) for e in train])