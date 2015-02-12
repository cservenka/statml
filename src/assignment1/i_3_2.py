import os
import operator
import numpy as np

from i_3_1 import *

def splitList(lst, n):
    n = max(1, n)
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def nFoldCrossValidation(n, k, train):
	accuracy = []
	groups = splitList(train, len(train) / n)
	for i in range(n):
		rest = reduce(operator.add, groups[0:i]+groups[i+1:len(groups)])
		skip = groups[i]
		accuracy.append(getAccuracy(getSpecies(skip), [getMajority(getNeighbors(rest, e, k)) for e in skip]))
	return np.mean(accuracy)

if __name__ == '__main__':
	train = readFile(os.path.dirname(__file__) + '/../../data/IrisTrain2014.dt')
	test = readFile(os.path.dirname(__file__) + '/../../data/IrisTest2014.dt')

	results = []
	for i in range(1, 27, 2):
		results.append((i, nFoldCrossValidation(5, i, train)))
	
	kBest = max(results, key = operator.itemgetter(1))[0]

	print "Test set K_best = %d Accuracy = %s" % (kBest, getAccuracy(getSpecies(test), [getMajority(getNeighbors(train, e, kBest)) for e in test]))