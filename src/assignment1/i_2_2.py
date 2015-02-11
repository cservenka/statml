import numpy as np
import matplotlib.pyplot as mpl

# The parameters of the Gaussian distribution.
mu = np.array([1,2]).T
Sigma = np.array([[0.3, 0.2], [0.2, 0.2]])

def generateSamples(N, mu, Sigma):
	# Compute the Cholesky decomposition of Sigma.
	L = np.linalg.cholesky(Sigma)
	
	# List to hold the samples.
	samples = []
	
	for _ in xrange(N):
		# Draw a 2D point from the zero mean unit variance
		# Gaussian distribution, and use it to generate a
		# sample (using the formula in the assignment text).
		sample = mu + np.dot(L, np.random.randn(2))
		samples.append(sample)
	
	return np.array(samples)

# Number of samples.
N = 100

# The data set is a list (or a numpy array) of points.	
dataset = generateSamples(N, mu, Sigma)

if __name__ == '__main__':
	# Plot the data set. When plotting, we must transform our list
	# of pairs into a pair of lists (x and y lists). Therefore, we
	# transpose the data set.
	mpl.plot(dataset.T[0,], dataset.T[1,], 'ro')
	mpl.show()