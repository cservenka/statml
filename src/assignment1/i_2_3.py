from __future__ import division
import i_2_2
from i_2_2 import mu

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
	mpl.plot(dataset.T[0,], dataset.T[1,], 'ro', label="Dataset")
	
	# Compute the sample mean and plot it.
	sampleMean = muML(dataset)
	mpl.plot(sampleMean[0], sampleMean[1], 'bo', label="Sample mean")
	string = "("+str(sampleMean[0])+", "+str(sampleMean[1])+")"
	mpl.annotate(string, xy=(sampleMean[0], sampleMean[1]), xytext=(75,-125), 
                 textcoords='offset points', ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                                 color='blue'))
	
	# Plot distribution mean.
	mpl.plot(mu[0], mu[1], 'mo', label="Distribution mean")
	string = "("+str(mu[0])+", "+str(mu[1])+")"
	mpl.annotate(string, xy=(mu[0], mu[1]), xytext=(50,-100), 
                 textcoords='offset points', ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                                 color='purple'))
	
	mpl.title('Sample with sample mean and distribution mean ')
	mpl.ylabel('y')
	mpl.xlabel('x')
	mpl.legend(loc=2)
	mpl.show()

if __name__ == '__main__':
	(dataset, _) = i_2_2.run()
	run(dataset)