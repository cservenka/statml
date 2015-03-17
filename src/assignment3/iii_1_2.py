from __future__ import division

from iii_1_1 import NeuralNetwork, readFile, transposeList
import os
import math
import operator
import numpy as np
import matplotlib.pyplot as mpl

def sinc(x):
    return math.sin(x) / x

# Flatten a list of lists to a single list.
def flatten(lst):
    return [item for sublist in lst for item in sublist]

# Predicate that says whether or not the gradient is small enough to stop
# training.
def smallEnough(derivatives):
    # Neutral value of &.
    result = True
    
    # Arbitrary epsilon value.
    epsilon = 0.01
    
    # All the partial derivatives must be smaller than epsilon.
    for deriv in derivatives.itervalues():
        result &= (math.fabs(deriv) < epsilon)
    
    return result

# Load training and test datasets.
train = readFile(os.path.dirname(__file__) + '/../../data/sincTrain25.dt')
test = readFile(os.path.dirname(__file__) + '/../../data/sincValidate10.dt')

# Train the network with the given parameters.
def runTraining(M, learning_rate):
    # One-dimensional inputs.
    D = 1
    
    # One-dimensional output.
    K = 1
    
    # Create neural network.
    nn = NeuralNetwork(D, M, K)
    
    # The training and test errors over the course of learning.
    train_errors = []
    test_errors = []
    
    # Run the network on the test data and compute test error.
    nn.load(test)
    nn.runNetwork()
    test_errors.append(nn.error)
    
    # Run the network on the training data. Compute errors and derivatives (used
    # in steepest descent).
    nn.load(train)
    nn.runNetwork()
    derivatives = nn.dE
    train_errors.append(nn.error)
    
    # Continue adjusting until the gradient is small enough.
    while not smallEnough(derivatives):
        # Apply steepest descent.
        for k in nn.w.iterkeys():
            nn.w[k] -= learning_rate * derivatives[k]
        
        # Run the network on the test data.
        nn.load(test)
        nn.runNetwork()
        test_errors.append(nn.error)
        
        # Run the network on the training data.
        nn.load(train)
        nn.runNetwork()
        derivatives = nn.dE
        train_errors.append(nn.error)
    
    return (nn, train_errors, test_errors)

def run():
    # Try with these numbers of hidden neurons.
    Ms = [2,4]
    # Try with these learning rates.
    learning_rates = [0.1, 0.5]
    
    # Different styles for the plots.
    train_colours = ['b','y']
    train_shapes = ['o','*']
    test_colours = ['c','r']
    test_shapes = ['^', 'v']
    
    # The training error when the training stops.
    final_train_errors = []
    
    for M in Ms:
        for i in range(len(learning_rates)):
            learning_rate = learning_rates[i]
            
            # Train the network with the current combination of parameters.
            (nn, train_errors, test_errors) = runTraining(M, learning_rate)
            
            # The last error obtained.
            final_train_errors.append((nn, train_errors[len(train_errors)-1]))
            
            # Plot the training errors over the course of learning.
            train_colour = train_colours[i]
            train_shape = train_shapes[i]
            mpl.plot(train_errors, train_colour+train_shape, label = 'training, LR = ' + str(learning_rate))
            
            # Plot the test errors over the course of learning.
            test_colour = test_colours[i]
            test_shape = test_shapes[i]
            mpl.plot(test_errors, test_colour+test_shape, label = 'test, LR = ' + str(learning_rate))
        
        # Make new plot for each value of M.
        mpl.title('M = ' + str(M))
        mpl.xlabel('steps')
        mpl.ylabel('mean-squared error')
        mpl.yscale('log')
        mpl.legend()
        mpl.show()
    
    print "Training errors obtained"
    print map(lambda (a,b) : b, final_train_errors)
    
    # The network that obtained the best training error.
    best_nn = min(final_train_errors, key = operator.itemgetter(1))[0]
    
    # The x-values to plot for.
    xs = np.arange(-10.0, 10.0, 0.05).tolist()
    # Targets are not used for anything since we are not training anymore. But
    # the network algorithm requires there to be target values.
    targets = map(lambda x : 1.0, xs)
    # Dataset consisting of the x-values to plot and dummy target values.
    dataset = transposeList([xs, targets])
    
    # Apply the real sinc function to the x-values.
    realResults = map(lambda x : sinc(x), xs)
    
    # Load the x-values into the network and run the network.
    best_nn.load(dataset)
    best_nn.runNetwork()
    computedResults = flatten(best_nn.outputs)
    
    # Plot the real sinc function and the sinc values computed using the network.
    mpl.plot(xs, realResults, 'bo', label = 'real')
    mpl.plot(xs, computedResults, 'y*', label = 'computed')
    mpl.title('sinc function')
    mpl.xlabel('x')
    mpl.ylabel('y')
    mpl.legend()
    mpl.show()

if __name__ == '__main__':
    run()