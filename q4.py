import my_library as ml
import math
#charge density
lam = 1
#length of wire
l = 2
Q = l*lam
#total charge
epsilon = 8.8451 * 1e12
#charge of half of wire
q2 = Q/2
#distance from the wire
s = 1

def fun2(x):
    return q2/math.sqrt(s**2+ x**2)

a = -1
b = 1
degree = [4,5,6]

print('Actual potential = ', round(math.log((math.sqrt(2)+1)/(math.sqrt(2)-1))),9)

##For calculating the roots and weights of legendre polynomial i have used 'np.polynomial.legendre.leggauss()' instead of pasting the corresponding values given

for i in range(len(degree)):
    ans = ml.gaussian_quadrature(fun2,degree[i],a,b)
    print('Calculated Potential(for degree %s) = '%degree[i],round(ans,9))
#print(ml.gaussian_quadrature(fun2,5,-1,1))

'''
Potential(for degree 4) =  1.76205418
Potential(for degree 5) =  1.762855296
Potential(for degree 6) =  1.76273005

'''