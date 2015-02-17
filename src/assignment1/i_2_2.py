import numpy as np
import matplotlib.pyplot as mpl

# The parameters of the Gaussian distribution.
mu = np.array([1,2]).T
Sigma = np.array([[0.3, 0.2], [0.2, 0.2]])

# Draw a set of 2D points from the zero mean unit variance
# Gaussian distribution.
def generageZValues(N):
	zs = []
	for _ in xrange(N):
		z = np.random.randn(2)
		zs.append(z)
	return zs

# Use the random 2D points to generate samples (using the
# formula in the assignment text).
def generateSamples(zs, mu, Sigma):
	# Compute the Cholesky decomposition of Sigma.
	L = np.linalg.cholesky(Sigma)
	
	# List to hold the samples.
	samples = []
	
	for z in zs:
		sample = mu + np.dot(L, z)
		samples.append(sample)
	
	return np.array(samples)

def run():
	# Number of samples.
	N = 100
	
	# Generate random 2D points.
	zs = generageZValues(N)
	
	# Generate samples using the random 2D points.
	dataset = generateSamples(zs, mu, Sigma)
	
	# Plot the data set. When plotting, we must transform our list
	# of pairs into a pair of lists (x and y lists). Therefore, we
	# transpose the data set.
	mpl.plot(dataset.T[0,], dataset.T[1,], 'ro', label="Gaussian distribution")
	
	mpl.title('Sample from 2D gaussian distribution')
	mpl.ylabel('y')
	mpl.xlabel('x')
	mpl.legend()
	mpl.show()
	
	# Return the data set and the random 2D points for later use.
	return (dataset, zs)

if __name__ == '__main__':
	run()