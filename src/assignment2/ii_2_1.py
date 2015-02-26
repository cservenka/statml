import os.path
import numpy as np
import matplotlib as mpl

def readFile(filename):
    #Interpret first two columns as floats, last column as int
    #toNum = lambda (x1,x2,x3,x4,x5,t) : float(b) if a == 2 else float(b)

    #List of lists, each containing 2 floats and 1 int
    return [map(float, line.strip().split(' ')) for line in open(filename)]

def get_design_matrix(xs,M):
    lst = []
    for i in xrange(len(xs)):
        xi = xs[i]
        lst.append([1.0] + xi)
    return np.matrix(lst)

def get_w_ML(phi,t):
    return np.linalg.pinv(phi)*t

def y(x,w,D):
    return w[0] + sum([w[i]*x[i] for i in range(1,D,1)])

def selection(dataset, start, end):
    return transposeList(transposeList(dataset)[start:(end+1)])

def transposeList(lst):
    return map(list, zip(*lst))

def sel1(dataset):
    return (selection(dataset, 2, 3), 2)
    
def sel2(dataset):
    return (selection(dataset, 4, 4), 1)

def sel3(dataset):
    return (selection(dataset, 0, 4), 5)

def run():
    train = readFile(os.path.dirname(__file__) + '/../../data/sunspotsTrainStatML.dt')
    test = readFile(os.path.dirname(__file__) + '/../../data/sunspotsTestStatML.dt')
    
    (xs, M) = sel1(train)
    print xs
    t = selection(train, 5, 5)
    phi = get_design_matrix(xs, M)
    w = get_w_ML(phi, t)
    for x in xs:
        print y(x,w,M)
    
if __name__ == '__main__':
    run()