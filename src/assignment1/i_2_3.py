from __future__ import division
import i_2_2

import matplotlib.pyplot as mpl

# Compute the maximum likelihood sample mean of
# a given data set.
def muML(dataset):
	N = len(dataset)
	n = 0
	for x in dataset:
		n += x
	return (1/N) * n

def run(dataset):
	# Plot the data set.
	mpl.plot(dataset.T[0,], dataset.T[1,], 'ro')
	
	# Compute the sample mean and plot it.
	mean = muML(dataset)
	mpl.plot(mean[0], mean[1], 'bo')
	
	mpl.ylabel('Sample together with maximum likelihood mean (in blue)')
	mpl.show()

if __name__ == '__main__':
	(dataset, _) = i_2_2.run()
	run(dataset)