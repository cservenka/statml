import os
import operator
import numpy as np

from i_3_1 import readFile, getAccuracy

# Split lst into chunks of size n.
def splitList(lst, n):
    n = max(1, n)
    return [lst[i:i + n] for i in range(0, len(lst), n)]

# Perform n-fold cross validation of the k-NN classifier on the train data set.
def nFoldCrossValidation(n, k, train):
    accuracy = []
    groups = splitList(train, len(train) / n)
    for i in range(n):
        rest = reduce(operator.add, groups[0:i]+groups[i+1:len(groups)])
        skip = groups[i]
        accuracy.append(getAccuracy(k, rest, skip))
    return np.mean(accuracy)

# Run the cross validation for several different values of k to determine the best one.
def determineBestK(train):
    results = []
    for i in range(1, 27, 2):
        result = nFoldCrossValidation(5, i, train)
        print "i = %d: Validation-result = %s" % (i, result)
        #results.append((i, nFoldCrossValidation(5, i, train)))
        results.append((i, result))
    
    kBest = max(results, key = operator.itemgetter(1))[0]
    
    return kBest

def run():
    train = readFile(os.path.dirname(__file__) + '/../../data/IrisTrain2014.dt')
    test = readFile(os.path.dirname(__file__) + '/../../data/IrisTest2014.dt')
    
    # Determine the best k.
    kBest = determineBestK(train)
    print "k_best = %d" % kBest
    
    print "Training set: Accuracy = %s" % getAccuracy(kBest, train, train)
    print "Test set: Accuracy = %s" % getAccuracy(kBest, train, test)

if __name__ == '__main__':
    run()