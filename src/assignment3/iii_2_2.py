import os
import operator
import numpy as np
import itertools
from sklearn import svm
from iii_2_1 import normalize, readFile

# Split lst into chunks of size n.
def splitList(lst, n):
	n = max(1, n)
	return [lst[i:i + n] for i in range(0, len(lst), n)]

def getLoss(parameters, test, train):
	(C, gamma) = parameters
	classifier = svm.SVC(C = C, kernel = 'rbf')
	classifier.gamma = gamma

	result_train = [a[-1] for a in train]
	data_train = [a[:-1] for a in train]
	
	result_test = [a[-1] for a in test]
	data_test = [a[:-1] for a in test]

	classifier.fit(data_train, result_train)
	result_classifier = classifier.predict(data_test)

	loss = lambda (p, r): 0 if p == r else 1
	return np.mean(map(loss, zip(result_test, result_classifier)))

def nFoldCrossValidation(n, data, parameters):
	accuracy = []
	groups = splitList(data, len(data) / n)
	for i in range(n):
		rest = reduce(operator.add, groups[0:i]+groups[i+1:len(groups)])
		skip = groups[i]
		accuracy.append(getLoss(parameters, skip, rest))
	return np.mean(accuracy)

def hyperparameterSelection(data):
	C_range = [0.1, 1, 10, 100, 1000]
	gamma_range = [0.001, 0.01, 0.1, 1, 10]
	hyperparameter_grid = [pair for pair in itertools.product(C_range, gamma_range)]

	results = []
	for parameters in hyperparameter_grid:
		result = nFoldCrossValidation(5, data, parameters)
		results.append((result, parameters))

	return min(results, key = operator.itemgetter(0))

def run():
	train = readFile(os.path.dirname(__file__) + '/../../data/parkinsonsTrainStatML.dt')
	test = readFile(os.path.dirname(__file__) + '/../../data/parkinsonsTestStatML.dt')

	normalized_train = normalize(train)
	normalized_test  = normalize(test)

	(train_loss, train_params) = hyperparameterSelection(train)
	(normalized_train_loss, normalized_train_params) = hyperparameterSelection(normalized_train)
	
	train_accuracy = 1 - train_loss
	test_accuracy  = 1 - getLoss(train_params, test, train)
	normalized_train_accuracy = 1 - normalized_train_loss
	normalized_test_accuracy  = 1 - getLoss(normalized_train_params, normalized_test, normalized_train)

	print "Training set: Hyperparameter config = (%s, %s)" % (train_params[0], train_params[1])
	print "Training set: Accuracy = %s" % train_accuracy
	print "Test set: Accuracy = %s" % test_accuracy

	print "Normalized Training set: Hyperparameter config = (%s, %s)" % (normalized_train_params[0], normalized_train_params[1])   
	print "Normalized Training set: Accuracy = %s" % normalized_train_accuracy
	print "Normalized Test set: Accuracy = %s" % normalized_test_accuracy
	
if __name__ == '__main__':
	run()
