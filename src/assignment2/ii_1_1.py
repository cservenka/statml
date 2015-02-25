import os.path

def readFile(filename):
	#Interpret first two columns as floats, last column as int
	toNum = lambda (a, b) : int(b) if a == 2 else float(b)

	#List of lists, each containing 2 floats and 1 int
	return [map(toNum, enumerate(line.strip().split(' '))) for line in open(filename)]

def run():
	train = readFile(os.path.dirname(__file__) + '/../../data/IrisTrain2014.dt')
	test = readFile(os.path.dirname(__file__) + '/../../data/IrisTest2014.dt')
	

	# http://scikit-learn.org/stable/modules/multiclass.html

	
if __name__ == '__main__':
	run()