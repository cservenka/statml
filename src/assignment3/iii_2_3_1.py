import os
from sklearn import svm
from iii_2_1 import normalize, readFile

C = 1
gamma = 0.01
test = readFile(os.path.dirname(__file__) + '/../../data/parkinsonsTestStatML.dt')

classifier = svm.SVC(C = C, kernel = 'rbf')
classifier.gamma = gamma

result_test = [a[-1] for a in test]
data_test = [a[:-1] for a in test]

result = classifier.fit(data_test, result_test)

coef = result.dual_coef_[0].tolist()

n_bound = coef.count(C) + coef.count(-C)
n_free = len(coef) - n_bound

print "bounded: %d \t free: %d" % (n_bound, n_free)


def run():
	print "Hello, world"

if __name__ == '__main__':
    run()


