import numpy as np
import matplotlib.pyplot as mpl

mu = np.array([1,2]).T
Sigma = np.array([[0.3, 0.2], [0.2, 0.2]])
L = np.linalg.cholesky(Sigma)
z = np.random.randn(2)

x = []
y = []
for i in xrange(100):
	temp = mu + np.dot(L, np.random.randn(2))
	x.append(temp[0])
	y.append(temp[1])

l = np.array([np.array(x), np.array(y)])

if __name__ == '__main__':
	mpl.plot(l[0,], l[1,], 'ro')
	mpl.show()