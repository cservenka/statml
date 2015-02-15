from __future__ import division
import i_2_2
from i_2_2 import generateSamples, mu, Sigma
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

# Rotate the covariance matrix Sigma by theta degrees.
def rotate(Sigma, theta):
	theta_radian = theta * np.pi / 180.
	R_theta = np.array([[np.cos(theta_radian), -np.sin(theta_radian)],
					[np.sin(theta_radian), np.cos(theta_radian)]])
	return np.dot(np.dot(np.linalg.inv(R_theta), Sigma), R_theta)

def drawEigenVectors(dataset):
	# Compute the sample covariance matrix.
	covariance = SigmaML(dataset)
	
	# Compute the eigenvalues and eigenvectors of the covariance matrix.
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

def run(dataset, zValues):
	# Plot the data set.
	mpl.plot(dataset.T[0,], dataset.T[1,], 'ro')
	
	mpl.ylabel('Sample together with eigenvectors')
	
	# Draw the eigenvectors of the covariance matrix on top of the data set.
	drawEigenVectors(dataset)
	
	# Create a new figure.
	mpl.figure()

	# Angles to rotate by.
	angles = [30,60,90]
	shapes = ['o','^','s']
	colours = ['b','c','r']

	# For each angle, rotate the covariance matrix and generate a new sample (using the
	# same z-values as earlier).
	for i in range(len(angles)):
		Sigma_theta = rotate(Sigma, angles[i])
		new_dataset = generateSamples(zValues, mu, Sigma_theta)
		mpl.plot(new_dataset.T[0,], new_dataset.T[1,], colours[i]+shapes[i])
	
	mpl.ylabel('Sample rotated 30, 60, and 90 degrees')
	mpl.show()

if __name__ == '__main__':
	(dataset, zValues) = i_2_2.run()
	run(dataset, zValues)