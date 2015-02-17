from __future__ import division

import numpy as np
import matplotlib.pyplot as mpl
import math

# Compute the y-value of the univariate Gaussian density function
# (with the given mu and sigma parameters) corresponding to the
# given x-value.
def gauss(mu, sigma, x):
	return (1 / (2 * math.pi * (sigma ** 2)) ** (1 / 2)) * \
		np.exp(-1 / (2 * (sigma ** 2)) * ((x - mu) ** 2))

def run():
	# The range of x-values to plot.
	xs = np.arange(-10.0, 10.0, 0.1)
	
	# The three different pairs of mu and sigma values to use.
	mus = [-1,0,2]
	sigmas = [1,2,3]
	
	# Make a plot for each pair of parameters.
	for i in range(len(mus)):
		mu = mus[i]
		sigma = sigmas[i]
		mpl.plot(xs, gauss(mu,sigma,xs), label="Mu = "+str(mu)+", sigma = "+str(sigma))
	
	mpl.title('Gaussian distributions')
	mpl.ylabel('y')
	mpl.xlabel('x')
	mpl.legend()
	mpl.show()

if __name__ == '__main__':
	run()