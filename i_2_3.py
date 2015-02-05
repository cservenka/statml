from __future__ import division
import i_2_2

import numpy as np
import matplotlib.pyplot as mpl

def uml(lst):
	N = len(lst)
	n = 0
	for x in lst:
		n += x
	return (1/N) * n

ml = uml(i_2_2.l.T)
mpl.plot(ml[0], ml[1], 'bo')

mpl.show()

