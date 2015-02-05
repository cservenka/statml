from __future__ import division
from assignment1.i_2_2 import *
from assignment1.i_2_3 import *

import numpy as np
import matplotlib.pyplot as mpl

def sml(lst):
	tmp = uml(lst)
	N = len(lst)
	n = 0
	for x in lst:
		n += np.outer((x - tmp), (x - tmp).T)
	return (1 / N) * n

if __name__ == '__main__':
	mpl.plot(l[0,], l[1,], 'ro')
	
	a = sml(l.T)
	(lam, e) = np.linalg.eig(a)

	for (index, li) in enumerate(lam):
		tmp = mu + np.sqrt(li) * e[index]
		mpl.plot(tmp[0], tmp[1], 'bo')

	mpl.show()