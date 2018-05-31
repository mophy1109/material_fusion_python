import numpy
from scipy.optimize import root

def F(x):
    L = numpy.zeros(8)
    L[0] = x[0]*x[0] - x[0]
    L[1] = x[1]*x[1] - x[1]
    L[2] = x[2]*x[2] - x[2]
    L[3] = x[3]*x[3] - x[3]
    L[4] = x[4]*x[4] - x[4]
    L[5] = x[5]*x[5] - x[5]
    L[6] = x[6]*x[6] - x[6]
    L[7] = x[7]*x[7] - x[7]

    L[8] = x[0]+x[1]+x[2]-2
    L[9] = x[5]+x[6]-x[7]-2
    L[10] = 3*x[0]+x[1]+x[2]-1
    L[11] = x[0]+x[4]+x[3]-2
    L[12] = x[3]-2*x[5]+x[7]-2
    return L

x = [1,1,1,1,1,1,1,1]

print x
print F(x)
y = root(F, x)
print y.x, y.fun