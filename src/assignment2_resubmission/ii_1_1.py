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
def getDelta_k(x, mu_k, sigma, prior):
	x = np.matrix(x)
	mu_k = np.matrix(mu_k)
	sigma = np.matrix(sigma)
	sig_inv = np.linalg.inv(sigma)

	return (x*sig_inv*mu_k.T - 0.5*mu_k*sig_inv*mu_k.T + np.log(prior)).item(0)

# Returns mu_k for a given class
def getMu_k(s_k):
	return (1 / len(s_k)) * np.sum(s_k, axis=0)

# Returns Sigma
def getSigma(dataset, splitset):
	l = len(dataset)
	m = len(set(getSpecies(dataset)))
	res = 0
	for k in xrange(m):
		mu_k = getMu_k(splitset[k])
		res_k = 0
		for i in splitset[k]:
			res_k += np.outer((i - mu_k), (i-mu_k).T) 
		res += res_k
	return (1 / (l - m)) * res

# Returns the prior_k
def getPrior_k(splitset, dataset):
	return len(splitset) / len(dataset)

# Groups a dataset by its species
def split(dataset, m):
	res = [[] for _ in xrange(m)]
	for elem in dataset:
		res[int(elem[-1])].append(elem[:-1])
	return np.array(res)

#Get list of species (as int) from a data set
def getSpecies(dataset):
	return [e[-1] for e in dataset]

# Classifies a dataset using Linear discriminant analysis
def classify(dataset):
	res = []
	m = len(set(getSpecies(dataset)))
	splitset = split(dataset, m)
	sigma = getSigma(dataset, splitset)
	for x in dataset:
		lst = []
		for k in xrange(m):
			mu = getMu_k(splitset[k])
			prior = getPrior_k(splitset[k], dataset)
			lst.append((k, getDelta_k(x[:-1], mu, sigma, prior)))
		res.append((x, max(lst, key=operator.itemgetter(1))[0]))
	return res

#Get 1 minus the mean of the 0-1 loss for each data entry
def getAccuracy(dataset):
	actual = getSpecies(dataset)
	predicted = [x[1] for x in classify(dataset)]
	loss = lambda (p, r): 0 if p == r else 1
	return 1 - np.mean(map(loss, zip(actual, predicted)))

def run():
	train = readFile(os.path.dirname(__file__) + '/../../data/IrisTrain2014.dt')
	test = readFile(os.path.dirname(__file__) + '/../../data/IrisTest2014.dt')

	# Perform LDA and print accuracy results for both sets
	print "Accuracy of LDA on training dataset: %s" % getAccuracy(train)
	print "Accuracy of LDA on test dataset: %s" % getAccuracy(test)
	
if __name__ == '__main__':
	run()
