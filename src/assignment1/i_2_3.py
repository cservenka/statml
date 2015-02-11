from __future__ import division
from i_2_2 import dataset

import matplotlib.pyplot as mpl

# Compute the maximum likelihood sample mean of
# a given data set.
def muML(dataset):
	N = len(dataset)
	n = 0
	for x in dataset:
		n += x
	return (1/N) * n

if __name__ == '__main__':
	# Plot the data set.
	mpl.plot(dataset.T[0,], dataset.T[1,], 'ro')
	
	# Get the sample mean and plot it.
	mean = muML(dataset)
	mpl.plot(mean[0], mean[1], 'bo')
	
	mpl.show()