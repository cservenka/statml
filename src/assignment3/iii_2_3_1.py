import os
from sklearn import svm
from iii_2_1 import normalize, readFile

def run():
	test = readFile(os.path.dirname(__file__) + '/../../data/parkinsonsTestStatML.dt')
	gamma = 0.01
	result_test = [a[-1] for a in test]
	data_test = [a[:-1] for a in test]

	for C in [0.1, 1, 10, 100, 1000]:
		classifier = svm.SVC(C = C, kernel = 'rbf')
		classifier.gamma = gamma

		result = classifier.fit(data_test, result_test)
		
		coef = result.dual_coef_[0].tolist()
		n_bound = coef.count(C) + coef.count(-C)
		n_free = len(coef) - n_bound

		print "C = %.1f \t bounded support vectors: %d \t free support vectors: %d" % (C, n_bound, n_free)

if __name__ == '__main__':
    run()