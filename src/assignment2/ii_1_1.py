from __future__ import division
import os.path
import numpy as np
import operator


def readFile(filename):
	#Interpret first two columns as floats, last column as int
	toNum = lambda (a, b) : int(b) if a == 2 else float(b)

	#List of lists, each containing 2 floats and 1 int
	return [map(toNum, enumerate(line.strip().split(' '))) for line in open(filename)]

# Returns delta_k
def get_delta_k(x, mu_k, sigma, prior):
	x = np.matrix(x)
	mu_k = np.matrix(mu_k)
	sigma = np.matrix(sigma)
	sig_inv = np.linalg.inv(sigma)

	return (x*sig_inv*mu_k.T - 0.5*mu_k*sig_inv*mu_k.T + np.log(prior)).item(0)

# Returns Mu for a given class
def get_mu_k(s_k):
	return (1 / len(s_k)) * np.sum(s_k, axis=0)

# Returns Sigma
def get_sigma(dataset, splitset):
	l = len(dataset)
	m = 3
	res = 0
	for k in xrange(m):
		mu_k = get_mu_k(splitset[k])
		res_k = 0
		for i in splitset[k]:
			res_k += np.outer((i - mu_k), (i-mu_k).T) 
		res += res_k
	return (1 / (len(dataset) - m)) * res

# Returns the prior
def get_prior_k(splitset, dataset):
	return len(splitset) / len(dataset)

# Groups a dataset by its species
def split(dataset, m):
	res = [[] for x in xrange(m)]
	for elem in dataset:
		res[elem[-1]].append(elem[:-1])
	return np.array(res)

# Classifies a dataset using Linear discriminant analysis
def classify(dataset):
	res = []
	splitset = split(dataset, 3)
	sigma = get_sigma(dataset, splitset)
	for x in dataset:
		lst = []
		for k in xrange(3):
			mu = get_mu_k(splitset[k])
			prior = get_prior_k(splitset[k], dataset)
			lst.append((k, get_delta_k(x[:-1], mu, sigma, prior)))
		res.append((x, max(lst, key=operator.itemgetter(1))[0]))
	return res

#Get list of species (as int) from a data set
def getSpecies(dataset):
	return [e[-1] for e in dataset]

#Get 1 minus the mean of the 0-1 loss for each data entry
def getAccuracy(dataset):
	actual = getSpecies(dataset)
	predicted = [x[1] for x in classify(dataset)]
	loss = lambda (p, r): 0 if p == r else 1
	return 1 - np.mean(map(loss, zip(actual, predicted)))

def run():
	train = readFile(os.path.dirname(__file__) + '/../../data/IrisTrain2014.dt')
	test = readFile(os.path.dirname(__file__) + '/../../data/IrisTest2014.dt')

	print "Accuracy of LDA on training dataset: " + str(getAccuracy(train))
	print "Accuracy of LDA on test dataset: " + str(getAccuracy(test))
	
if __name__ == '__main__':
	run()