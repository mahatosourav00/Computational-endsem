import my_library as ml
import matplotlib.pyplot as plt
import numpy as np

F = ml.matrix_read("esem4fit.txt",'r')

F = ml.transpose(F)
#print(F)

X = F[0]
#print(X)

Y = F[1]
#print(Y)

sig = [1 for i in range(len(X))]

def phi0(x):
    return 1

def phi1(x):
    return x

def phi2(x):
    return (3*x**2-1)/2

def phi3(x):
    return (5*x**3-3*x)/2

def phi4(x):
    return (35*x**4-30*x**2+3)/8

def phi5(x):
    return (63*x**5-70*x**3+15*x)/8

def phi6(x):
    return (231*x**6-315*x**4+105*x**2-5)/16

phi = [phi0, phi1, phi2, phi3, phi4, phi5, phi6]


fit,covar = ml.polynomial_fit_chebyshev(X,Y,sig,phi)
    
print("Fitting parameters are: ",fit)
print("Covariance matrix is: ")
ml.matrix_print(covar)



Yn = ml. Chebyshev_fun(X,phi,fit)

#chi square calculation
chi2 = 0
sum = 0
for i in range(len(Yn)):
    chi2 = chi2 + (Y[i] - Yn[i])**2/Yn[i]
condition_no = np.linalg.cond(covar, 1)
print("Condition no.: ",condition_no)
print('Chi squre = ', chi2)


plt.plot(X, Y,'go--',label = 'data' )
plt.plot(X, Yn, label = 'Fit')
plt.plot(label = 0 )
plt.legend()
plt.savefig('plot_q2.png')
plt.show()



'''
Fitting parameters are:  [[0.07003196671971398], [0.004301685837864321], [-0.010166710608800473], [0.013083743602879212], [0.11411855049286529], [-0.006726972223322476], [-0.0123845597126462]]
Covariance matrix is: 
0.03939854048321055  -4.131841312581797e-18  -0.0031667887846150984  -1.3948347218858502e-18  -0.006384612144582584  1.8556244505199504e-18  -0.010822673240747675  
-4.131841312581797e-18  0.11295125483436533  1.1763908584514444e-18  -0.017670875123743074  -6.708740828403796e-18  -0.031267370411183625  3.80955517796947e-18  
-0.0031667887846150984  1.1763908584514448e-18  0.18339837439276196  -1.997057477855786e-20  -0.03315001603861298  -3.8286720533946035e-19  -0.05541555705613892  
-1.3948347218858506e-18  -0.01767087512374307  -1.9970574778557882e-20  0.23609721162098654  1.3803501930733144e-18  -0.07647796503459686  -3.526765622523663e-20  
-0.006384612144582583  -6.708740828403796e-18  -0.03315001603861297  1.3803501930733142e-18  0.2952232434250448  1.7417705531109183e-18  -0.10488597094600074  
1.85562445051995e-18  -0.031267370411183625  -3.8286720533946044e-19  -0.07647796503459686  1.7417705531109183e-18  0.31038135160144803  -1.2501086674613357e-18  
-0.010822673240747677  3.809555177969469e-18  -0.05541555705613892  -3.5267656225236623e-20  -0.10488597094600072  -1.2501086674613357e-18  0.35821711388661465  

Condition no.:  15.589839766174608
Chi squre =  0.03766471126646097

'''