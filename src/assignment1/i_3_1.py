import numpy as np

def readFile(filename):
	lists = [line.strip().split(' ') for line in open(filename)]
	return np.array(lists)

if __name__ == '__main__':
	x = readFile('/home/asbjorn/Dropbox/Datalogi/Kandidat/StatML/Assignment1/IrisTrain2014.dt')
	print x
	print x.T