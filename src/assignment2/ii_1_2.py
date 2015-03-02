from __future__ import division

from ii_1_1 import readFile, getAccuracy

import os
import numpy as np

# Compute the maximum likelihood sample mean of
# a given data set.
def muML(dataset):
    N = len(dataset)
    n = 0
    for x in dataset:
        n += x
    return (1/N) * n

# Compute the maximum likelihood sample covariance of
# a given data set.
def sigmaML(dataset):
    mean = muML(dataset)
    N = len(dataset)
    n = 0
    for x in dataset:
        n += (x - mean) ** 2
    return (1 / N) * n

# Normalize a single feature by a given mean and variance.
def normalizeList(lst, mean, variance):
    # For each data point, subtract the mean and divide by the
    # square-root of the variance (i.e., standard deviation).
    return map(lambda x: (x-mean)/np.sqrt(variance), lst)

# Transpose a list of lists.
def transpose(data):
    # Temporarily use a numpy matrix.
    return np.array(data).T.tolist()

# Normalize a data set using the given set of parameters (mean and variance).
def normalizeDataset(dataset, parameters):
    # Transpose the data set so that we can extract the
    # columns (features) separately.
    transposedDataset = transpose(dataset)
    
    normalizedDataset = []
    
    for i in range(2):
        # Get mean and variance parameters.
        currentParameters = parameters[i]
        normalizedDataset.append(normalizeList(transposedDataset[i],
                                               currentParameters[0],  # mean
                                               currentParameters[1])) # variance
    normalizedDataset.append(transposedDataset[2]) # list of classes
    
    # Transpose back to original form.
    return transpose(normalizedDataset)

# Get the mean and variance for each feature in the data set.
def getParameters(dataset):
    # Transpose the data set so that we can extract the
    # columns (features) separately.
    transposedDataset = transpose(dataset)
    
    # Result is [[mean1, variance1], [mean2, variance2]]
    results = []
    
    for i in range(2):
        # Extract feature.
        feature = transposedDataset[i]
        
        # Compute the sample mean and variance.
        mean = muML(feature)
        variance = sigmaML(feature)
        results.append([mean, variance])
    
    return results

def run():
    # Read the data sets.
    train = readFile(os.path.dirname(__file__) + '/../../data/IrisTrain2014.dt')
    test = readFile(os.path.dirname(__file__) + '/../../data/IrisTest2014.dt')
    
    # Use the training set to compute the normalization parameters.
    parameters = getParameters(train)
    
    # Normalize the sets.
    normalizedTrain = normalizeDataset(train, parameters)
    normalizedTest = normalizeDataset(test, parameters)
     
    # Transpose the sets (to extract features).
    transposedNormalizedTrain = transpose(normalizedTrain)
    transposedNormalizedTest = transpose(normalizedTest)
     
    # Compute mean and variance for both features of the normalized training set
    # (to verify zero mean and unit variance).
    for i in range(2):
        # Extract feature.
        feature = transposedNormalizedTrain[i]
         
        # Compute the sample mean and variance.
        mean = muML(feature)
        variance = sigmaML(feature)
         
        print "Training set, feature = %d: Mean = %s, variance = %s" % (i, mean, variance)
     
    # Compute mean and variance for both features of the normalized test set.
    for i in range(2):
        # Extract feature.
        feature = transposedNormalizedTest[i]
         
        # Compute the sample mean and variance.
        mean = muML(feature)
        variance = sigmaML(feature)
         
        print "Test set, feature = %d: Mean = %s, variance = %s" % (i, mean, variance)
    
    # Repeating exercise II.1.1 with the normalized data sets.
    print "Training set: Accuracy = %s" % getAccuracy(normalizedTrain)
    print "Test set: Accuracy = %s" % getAccuracy(normalizedTest)

if __name__ == '__main__':
    run()
