from __future__ import division
import numpy as np
import matplotlib.pyplot as mpl

def uml(lst):
	N = len(lst)
	n = 0
	for x in lst:
		n += x
	return (1/N) * n

my = np.array([1,2]).T
E = np.array([[0.3, 0.2], [0.2, 0.2]])
L = np.linalg.cholesky(E)
z = np.random.randn(2)


x = []
y = []
for i in xrange(100):
	temp = my + np.dot(L, np.random.randn(2))
	x.append(temp[0])
	y.append(temp[1])

l = np.array([np.array(x), np.array(y)])

mpl.plot(l[0,], l[1,], 'ro')
ml = uml(l.T)
mpl.plot(ml[0], ml[1], 'bo')

mpl.show()

