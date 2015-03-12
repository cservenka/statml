import os
import numpy as np
import itertools
from sklearn import svm
from iii_2_1 import readFile, normalize

# Split lst into chunks of size n.
def splitList(lst, n):
    n = max(1, n)
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def nFoldCrossValidation(n, data, result):
    accuracy = []
    groups = splitList(data, len(data) / n)
    for i in range(n):
        rest = reduce(operator.add, groups[0:i]+groups[i+1:len(groups)])
        skip = groups[i]
        accuracy.append(4) #TODO
    return np.mean(accuracy)

def hyperparameterSelection(data, result):
	C_range = [0.1, 1, 10, 100, 1000]
	gamma_range = [0.001, 0.01, 0.1, 1, 10]
	hyperparameter_grid = [pair for pair in itertools.product(C_range, gamma_range)]

	results = []
	for (gamma, C) in hyperparameter_grid:
		result = nFoldCrossValidation(5, data, result)
		results.append((result, gamma, C))

	return max(results, key = operator.itemgetter(0))[0]

(train_data, train_result) = readFile(os.path.dirname(__file__) + '/../../data/parkinsonsTrainStatML.dt')

#classifier = svm.SVC(C = 1.0, kernel = 'rbf')
#classifier.gamma = 1.0
#classifier.fit(data, result)
