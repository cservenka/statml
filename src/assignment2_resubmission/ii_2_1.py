import os.path
import numpy as np
import matplotlib.pyplot as mpl
import math

# Read dataset from file.
def readFile(filename):
    # List of lists, each containing six floats.
    return [map(float, line.strip().split(' ')) for line in open(filename)]

# Compute the design matrix for the given (subset of the) dataset.
def get_design_matrix(dataset):
    # Return the phi matrix as list of lists.
    phi = []
    for i in range(len(dataset)):
        phi.append([1.0] + dataset[i])
    return phi

# Compute the maximum likelihood estimate of w using formula 3.15 in Bishop.
def get_w_ML(phi,t):
    result = np.dot(np.linalg.pinv(np.matrix(phi)), np.array(t))
    return flatten(result.tolist())

# Compute our estimate of the target value using data point x and model
# parameters w.
def y(x,w):
    return w[0] + sum([w[i+1]*x[i] for i in range(len(x))])

# Assuming that dataset is a list of lists, select the columns
# from start to end (0-indexed), both included.
def selection(dataset, start, end):
    return transposeList(transposeList(dataset)[start:(end+1)])

# Select rows 3-4 from the dataset.
def sel1(dataset):
    return selection(dataset, 2, 3)

# Select row 5 from the dataset.    
def sel2(dataset):
    return selection(dataset, 4, 4)

# Select rows 1-5 from the dataset.
def sel3(dataset):
    return selection(dataset, 0, 4)

# Flatten a list of lists to a single list.
def flatten(lst):
    return [item for sublist in lst for item in sublist]

# Transpose a list of lists.
def transposeList(lst):
    return (np.array(lst).T).tolist()

# Plot the predicted and actual values.
def plotPredictedAndActual(title, xs, ys, ts):
    mpl.plot(xs, ys, 'ro', label="predicted")
    mpl.plot(xs, ts, 'b^', label="actual")
    mpl.title(title)
    mpl.ylabel('y')
    mpl.xlabel('x')
    mpl.legend()
    mpl.show()

# Compute the RMS error.
def computeRMS(xs, w, ts):
    return math.sqrt(sum([(ts[i] - y(xs[i],w))**2 for i in range(len(xs))])/len(xs))

# Plot years vs. sunspots.
def plotYearsVsSunspots(title, dataset, w, ts):
    startYear = 1916
    
    # The year numbers in the dataset, assuming consecutive years
    # starting from startYear.
    xs = [startYear+i for (i,_) in enumerate(dataset)]
    ys = [y(x,w) for x in dataset]
    
    # Plot the predicted and actual values.
    mpl.plot(xs, ys, 'ro', label="predicted")
    mpl.plot(xs, ts, 'b^', label="actual")
    
    # Connect each predicted and actual value by a line.
    for x1,y1,t1 in zip(xs, ys, ts):
        mpl.plot([x1,x1],[y1,t1],'k-')
    
    mpl.title(title)
    mpl.ylabel('y')
    mpl.xlabel('x')
    mpl.legend()
    mpl.show()

def run():
    # Read datasets.
    train = readFile(os.path.dirname(__file__) +
                     '/../../data/sunspotsTrainStatML.dt')
    test = readFile(os.path.dirname(__file__) +
                    '/../../data/sunspotsTestStatML.dt')
    
    # Retrieve appropriate columns from the training set.
    train1 = sel1(train)
    train2 = sel2(train)
    train3 = sel3(train)
    ts_train = flatten(selection(train, 5, 5))
    
    # Compute the design matrices.
    phi1 = get_design_matrix(train1)
    phi2 = get_design_matrix(train2)
    phi3 = get_design_matrix(train3)
    
    # Compute the model parameters.
    w1 = get_w_ML(phi1, ts_train)
    w2 = get_w_ML(phi2, ts_train)
    w3 = get_w_ML(phi3, ts_train)
    
    # Flatten the list of singleton lists (for plotting) and compute
    # the predicted value for each point.
    xs_train_plot = flatten(train2)
    ys_train_plot = [y(x,w2) for x in train2]
    
    # Plot the predicted and actual values for the training set.
    plotPredictedAndActual('Training set, selection 2',
                           xs_train_plot, ys_train_plot, ts_train)
    
    # Retrieve appropriate columns from the test set.
    test1 = sel1(test)
    test2 = sel2(test)
    test3 = sel3(test)
    ts_test = flatten(selection(test, 5, 5))
    
    # Flatten the list of singleton lists (for plotting) and compute
    # the predicted value for each point.
    xs_test_plot = flatten(test2)
    ys_test_plot = [y(x,w2) for x in test2]
    
    # Plot the predicted and actual values for the test set in a new figure.
    mpl.figure()
    plotPredictedAndActual('Test set, selection 2',
                           xs_test_plot, ys_test_plot, ts_test)
    
    # Print the RMS errors.
    print 'RMS:'
    RMS1 = computeRMS(test1, w1, ts_test)
    print 'Selection 1: ' + str(RMS1)
    RMS2 = computeRMS(test2, w2, ts_test)
    print 'Selection 2: ' + str(RMS2)
    RMS3 = computeRMS(test3, w3, ts_test)
    print 'Selection 3: ' + str(RMS3)
    
    # Plot years vs. sunspots.
    plotYearsVsSunspots('Test set, selection 1, years vs. sunspots',
                        test1, w1, ts_test)
    plotYearsVsSunspots('Test set, selection 2, years vs. sunspots',
                        test2, w2, ts_test)
    plotYearsVsSunspots('Test set, selection 3, years vs. sunspots',
                        test3, w3, ts_test)

if __name__ == '__main__':
    run()