from __future__ import division
from i_2_2 import dataset, generateSamples, N, mu, Sigma
from i_2_3 import muML

import numpy as np
import matplotlib.pyplot as mpl

# Compute the maximum likelihood sample covariance matrix of
# a given data set.
def SigmaML(dataset):
	mean = muML(dataset)
	N = len(dataset)
	n = 0
	for x in dataset:
		n += np.outer((x - mean), (x - mean).T)
	return (1 / N) * n

def rotate(Sigma, theta):
	theta_radian = theta * np.pi / 180.
	R_theta = np.array([[np.cos(theta_radian), -np.sin(theta_radian)],
					[np.sin(theta_radian), np.cos(theta_radian)]])
	return np.dot(np.dot(np.linalg.inv(R_theta), Sigma), R_theta)

if __name__ == '__main__':
	# Plot the data set.
	mpl.plot(dataset.T[0,], dataset.T[1,], 'ro')
	
	# Get the sample covariance matrix.
	covariance = SigmaML(dataset)
	
	# Get the eigenvalues and eigenvectors of the covariance matrix.
	(eigen_values, eigen_vectors) = np.linalg.eig(covariance)
	
	# Lists to hold the (coordinates of the) starting points and
	# the vectors themselves. 
	startXs = []
	startYs = []
	vectorXs = []
	vectorYs = []
	
	for (i, eigen_value) in enumerate(eigen_values):
		# Compute the scaled eigenvector.
		scaled = np.sqrt(eigen_value) * eigen_vectors[:,i]
		
		# The starting point of the vector is mu (this has the
		# same effect as translating by mu).
		startXs.append(mu[0])
		startYs.append(mu[1])
		
		# The vector to draw.
		vectorXs.append(scaled[0])
		vectorYs.append(scaled[1])
	
	# Draw the vectors.
	ax = mpl.gca()
	ax.quiver(startXs,startYs,vectorXs,vectorYs,angles='xy',scale_units='xy',scale=1)
	
	mpl.figure()
	
	angles = [30,60,90]
	shapes = ['o','^','s']
	colours = ['b','c','r']
	
	for i in range(len(angles)):
		Sigma_theta = rotate(Sigma, angles[i])
		new_dataset = generateSamples(N, mu, Sigma_theta)
		mpl.plot(new_dataset.T[0,], new_dataset.T[1,], colours[i]+shapes[i])
	
	mpl.show()