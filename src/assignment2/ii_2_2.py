from ii_2_1 import readFile, sel1, sel2, sel3, selection, get_w_ML, flatten, get_design_matrix, computeRMS
import os.path
import numpy as np
import matplotlib.pyplot as mpl

# Compute the maximum a posteriori estimate of w.
def get_w_MAP(phi, t, alpha):
    # Compute the number of parameters in the model, M. 
    M = max(map(len, phi))
    # Prior mean.
    m_zero = np.zeros(M)
    # Prior variance matrix.
    S_zero = alpha * np.identity(M)
    # Create numpy versions of phi and t.
    phi_matrix = np.array(phi)
    t_vector = np.array(t)
    # Compute S_N_inv from fromula 3.51 in Bishop.
    S_N_inv = np.linalg.inv(S_zero) + np.dot(phi_matrix.T, phi_matrix)
    # Compute m_N from formula 3.50 in Bishop.
    m_N = np.dot(np.linalg.inv(S_N_inv),
                 (np.dot(np.linalg.inv(S_zero), m_zero) +
                  np.dot(phi_matrix.T, t_vector)))
    return m_N.tolist()

def computeAndPlotRMS(title, xs_train, xs_test, ts_train, ts_test):
    # Compute the design matrix.
    phi = get_design_matrix(xs_train)
    
    # Try several different values for the prior precision parameter.
    alphas = np.arange(0.25, 40, 0.25).tolist()
    
    # Compute the maximum likelihood estimate for comparison.
    w_ML = get_w_ML(phi, ts_train)
    RMS_ML = computeRMS(xs_test, w_ML, ts_test)
    
    RMSs_ML = []
    RMSs_MAP = []
    
    for alpha in alphas:
        # Compute the MAP estimate of w.
        w_MAP = get_w_MAP(phi, ts_train, alpha)
        # Compute the RMS error.
        RMS = computeRMS(xs_test, w_MAP, ts_test)
        # Add RMS error for this value of alpha.
        RMSs_MAP.append(RMS)
        # The RMS error for the maximum likelihood estimate is always the
        # same since it doesn't depend on alpha.
        RMSs_ML.append(RMS_ML)
    
    # Plot the RMS errors for the MAP estimate against the different
    # alpha values.
    mpl.plot(alphas, RMSs_MAP, 'ro', label='MAP RMS')
    # Plot the RMS error for the ML estimate (constant line).
    mpl.plot(alphas, RMSs_ML, 'b^', label='ML RMS')
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
    
    # Retrieve appropriate columns from the test set.
    test1 = sel1(test)
    test2 = sel2(test)
    test3 = sel3(test)
    ts_test = flatten(selection(test, 5, 5))
    
    # Compute and plot the RMS errors.
    computeAndPlotRMS('Selection 1', train1, test1, ts_train, ts_test)
    computeAndPlotRMS('Selection 2', train2, test2, ts_train, ts_test)
    computeAndPlotRMS('Selection 3', train3, test3, ts_train, ts_test)
    
if __name__ == '__main__':
    run()