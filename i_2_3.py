from __future__ import division
from i_2_2 import *

import numpy as np
import matplotlib.pyplot as mpl

def uml(lst):
	N = len(lst)
	n = 0
	for x in lst:
		n += x
	return (1/N) * n

ml = uml(l.T)

if __name__ == '__main__':
	mpl.plot(l[0,], l[1,], 'ro')
	mpl.plot(ml[0], ml[1], 'bo')
	mpl.show()