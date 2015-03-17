from __future__ import division

import os
import numpy as np

# Reads a datafile containing 22 features and a label on each line
def readFile(filename):
    lines = [line.strip().split(' ') for line in open(filename)]
    lines_rotated = np.rot90(lines, 3)
    (features, classifications) = (np.rot90([map(float, line) for line in lines_rotated[:-1]]), map(int, lines_rotated[-1])[::-1])
    return [features[i].tolist() + [classifications[i]] for i in xrange(len(features))]

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
    return map(list, zip(*data))

# Normalize a data set using the given set of parameters (mean and variance).
def normalizeDataset(dataset, parameters):
    # Transpose the data set so that we can extract the
    # columns (features) separately.
    transposedDataset = transpose(dataset)
    
    normalizedDataset = []
    
    for i in range(22):
        # Get mean and variance parameters.
        currentParameters = parameters[i]
        normalizedDataset.append(normalizeList(transposedDataset[i],
                                               currentParameters[0],  # mean
                                               currentParameters[1])) # variance
    normalizedDataset.append(transposedDataset[22]) # list of classes
    
    # Transpose back to original form.
    return transpose(normalizedDataset)

# Get the mean and variance for each feature in the data set.
def getParameters(dataset):
    # Transpose the data set so that we can extract the
    # columns (features) separately.
    transposedDataset = transpose(dataset)
    
    # Result is [[mean1, variance1], [mean2, variance2]]
    results = []
    
    for i in range(22):
        # Extract feature.
        feature = transposedDataset[i]
        
        # Compute the sample mean and variance.
        mean = muML(feature)
        variance = sigmaML(feature)
        results.append([mean, variance])
    
    return results

# Returns a fully normalized dataset
def normalize(dataset):
    return normalizeDataset(dataset, getParameters(dataset))

def run():
    # Read the data sets.
    train = readFile(os.path.dirname(__file__) + '/../../data/parkinsonsTrainStatML.dt')
    test = readFile(os.path.dirname(__file__) + '/../../data/parkinsonsTestStatML.dt')
    
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
    for i in range(22):
        # Extract feature.
        feature = transposedNormalizedTrain[i]
         
        # Compute the sample mean and variance.
        mean = muML(feature)
        variance = sigmaML(feature)
         
        print "Training set, feature = %d: Mean = %s, variance = %s" % (i, mean, variance)
     
    # Compute mean and variance for both features of the normalized test set.
    for i in range(22):
        # Extract feature.
        feature = transposedNormalizedTest[i]
         
        # Compute the sample mean and variance.
        mean = muML(feature)
        variance = sigmaML(feature)
         
        print "Test set, feature = %d: Mean = %s, variance = %s" % (i, mean, variance)
    
if __name__ == '__main__':
    run()
