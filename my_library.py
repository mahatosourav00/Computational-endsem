import math
import random
from re import M
import time
import matplotlib.pyplot as plt
import numpy as np



def runge_kutta_4(x0, y0, z0, h, fun1, fun2, a, b):#, file):
    X = []
    Y = []
    Z = []
    x01 = x0
    y01 = y0
    z01 = z0
    #file.write("{:<15}{:<15}\n".format(x0, y0))
    while round(x0,6) < b:
        k1y = h * fun1(x0, y0, z0)
        k1z = h * fun2(x0, y0, z0)
        k2y = h * fun1((x0 + h/2), (y0 + k1y/2), (z0 + k1z/2))
        k2z = h * fun2((x0 + h/2), (y0 + k1y/2), (z0 + k1z/2))
        k3y = h * fun1((x0 + h/2), (y0 + k2y/2), (z0 + k2z/2))
        k3z = h * fun2((x0 + h/2), (y0 + k2y/2), (z0 + k2z/2))
        k4y = h * fun1((x0 + h), (y0 + k3y), (z0 + k3z))
        k4z = h * fun2((x0 + h), (y0 + k3y), (z0 + k3z))
        y = y0 + (1/6 * (k1y + (2 * k2y) + (2 * k3y) + k4y))
        z0 = z0 + (1/6 * (k1z + (2 * k2z) + (2 * k3z) + k4z))
        x0 = x0 + h
        y0 = y
        X.append(x0)
        Y.append(y0)
        Z.append(z0)

    x0 = x01
    y0 = y01
    z0 = z01
    while round(x0,6) > a:
        k1y = h * fun1(x0, y0, z0)
        k1z = h * fun2(x0, y0, z0)
        k2y = h * fun1((x0 + h/2), (y0 + k1y/2), (z0 + k1z/2))
        k2z = h * fun2((x0 + h/2), (y0 + k1y/2), (z0 + k1z/2))
        k3y = h * fun1((x0 + h/2), (y0 + k2y/2), (z0 + k2z/2))
        k3z = h * fun2((x0 + h/2), (y0 + k2y/2), (z0 + k2z/2))
        k4y = h * fun1((x0 + h), (y0 + k3y), (z0 + k3z))
        k4z = h * fun2((x0 + h), (y0 + k3y), (z0 + k3z))
        y = y0 - (1/6 * (k1y + (2 * k2y) + (2 * k3y) + k4y))
        z0 = z0 - (1/6 * (k1z + (2 * k2z) + (2 * k3z) + k4z))
        x0 = x0 - h
        y0 = y
        #print(x0)
        X.append(x0)
        Y.append(y0)
        Z.append(z0)
#
        
        #file.write("{:<15.6}{:<15.10}\n".format(x0, y0))
    return X,Y,Z






'''

def Legendre(n,x):
	x=np.array(x)
	if (n==0):
		return x*0+1.0
	elif (n==1):
		return x
	else:
		return ((2.0*n-1.0)*x*Legendre(n-1,x)-(n-1)*Legendre(n-2,x))/n
 

# Derivative of the Legendre polynomials
def DLegendre(n,x):
	x=np.array(x)
	if (n==0):
		return x*0
	elif (n==1):
		return x*0+1.0
	else:
		return (n/(x**2-1.0))*(x*Legendre(n,x)-Legendre(n-1,x))

# Roots of the polynomial obtained using Newton-Raphson method
def LegendreRoots(polyorder,tolerance=1e-20):
    if polyorder<2:
        err=1
    else:
        roots=[]
        print((polyorder/2 + 1))
        for i in range(1,int((polyorder/2) + 1)):
            x=np.cos(np.pi*(i-0.25)/(polyorder+0.5))
            error=10*tolerance
            iters=0
            while (error>tolerance) and (iters<1000):
                dx=-Legendre(polyorder,x)/DLegendre(polyorder,x)
                x=x+dx
                iters=iters+1
                error=abs(dx)
            roots.append(x)
            # Use symmetry to get the other roots
        roots=np.array(roots)
        if polyorder%2==0:
            roots=np.concatenate( (-1.0*roots, roots[::-1]) )
        else:
            roots=np.concatenate( (-1.0*roots, [0.0], roots[::-1]) )
        err=0 # successfully determined roots
    return [roots, err]
# Weight coefficients
def GaussLegendreWeights(polyorder):
	W=[]
	[xis,err]=LegendreRoots(polyorder)
	if err==0:
		W=2.0/( (1.0-xis**2)*(DLegendre(polyorder,xis)**2) )
		err=0
	else:
		err=1 # could not determine roots - so no weights
	return [xis, W, err]


'''




def gaussian_quadrature(fun, order, a, b):
    res = np.polynomial.legendre.leggauss(order)
    #res = GaussLegendreWeights(order)
    #print(res)
    roots = res[0]
    print((roots))
    weights = res[1]
    #err = res[2]
    #if err == 0:
    sum = 0
    for i in range(len(roots)):
        y = ((b-a)*0.5*roots[i])+((b+a)*0.5)
        wfy = weights[i]*fun(y)
        sum = sum + wfy
    ans = (b-a)*0.5*sum
    #else:
    #    ans=None
    return ans






def pde_explicit(L, delx, delt,funx, time):
    # u matrix
    n = int(L/delx)+1
    u0 = make_matrix(n,1)
    u0[-1][-1] = funx(L)
    u0[0][0] = funx(0)

    #A matrix
    A = make_matrix(n,n)
    alpha = delt/(delx**2)
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i == j:
                A[i][i] = 1 - (2*alpha)
            elif i == (j-1) or j == (i-1):
                A[i][j] = alpha
    i = 1
    while i*delt < time:
        unew = matrix_multiplication(A,u0)
        u0 = matrix_copy(unew)
        i = i+1
        print(i)
    return u0


def pde_implicit(L, delx, delt,funx, time):
    # u matrix
    n = int(L/delx)+1
    u0 = make_matrix(n,1)
    u0[-1][-1] = funx(L)
    u0[0][0] = funx(0)

    #A matrix
    A = make_matrix(n,n)
    alpha = delt/(delx**2)
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i == j:
                A[i][i] = 1 + (2*alpha)
            elif i == (j-1) or j == (i-1):
                A[i][j] = -alpha
    A = lu_inverse(A)
    i = 1
    while i*delt < time:
        unew = matrix_multiplication(A,u0)
        u0 = matrix_copy(unew)
        i = i+1
        print(i)
    return u0


    










def cholesky_solution(A,B):
    A1 = matrix_copy(A)
    A1, B = partial_pivot_inverse(A1, B)
    L = cholesky_decomposition(A1)
    print(L)
    LT = transpose(L)
    print(L)
    #LLT = matrix_multiplication(L,LT)
    #print(LLT)
    B = forward_substitution_cholesky(L, B)
    B = backward_substituition_cholesky(LT, B)
    return B




def cholesky_decomposition(A):
    CD = make_matrix(len(A),len(A))
    for i in range(len(A)):
        
        for j in range(i+1):
            sum = 0
            for k in range(i+1):
                sum = sum + CD[i][k]*CD[j][k]
            if i == j:
                CD[i][i] = math.sqrt(abs(A[i][i]-sum))
            else:
                CD[i][j] = (A[i][j]-sum)/CD[j][j]
            
    #for i in range(len(A)):
    #    for j in range(len(A[i])):
    #        if i < j:
    #            sum = 0
    #            for k in range(i):
    #                sum = sum + (CD[i][k]*CD[k][j])
    #                A[i][j] = (A[i][j]-sum)/CD[i][i]
    return CD

def forward_substitution_cholesky(L, B):

    #creat Y matrix
    Y = [[0] for i in range(len(L))]
    print('L',L)
    # calculate Y matrix
    for i in range(len(L)):
        sum = 0
        for j in range (i+1):
            print(Y[j][0],L[i][j])
            sum = sum + L[i][j] * Y[j][0]
            print(sum,i,j)
        Y[i][0] = (B[i][0] - sum)/L[i][i]
    print('Y',Y)
    #return the calculated Y matrix
    return (Y)



def backward_substituition_cholesky(LT, Y):
    # creat x matrix
    X = [[0] for i in range(len(LT))]

    # calculate x matrix
    for i in range(len(LT) - 1, -1, -1):
        sum = 0
        for j in range(len(LT) - 1, i - 1, -1):
            sum = sum + LT[i][j] * X[j][0]
        X[i][0] = (Y[i][0] - sum) / LT[i][i]
    print('x',X)
    # return the calculated x matrix
    return (X)




def monte_carlo_steinmetz_vol(a1,a2,b1,b2,c1,c2,circle1,circle2,N):

    X1 = []
    Y1 = []
    Z1 = []
    vol_box = (a2 - a1) * (b2 - b1) * (c2 - c1)
    ans = 0
    inte = 0

    for i in range(N):
        x = random_MLCG(a1,a2)
        y = random_MLCG(b1, b2)
        z = random_MLCG(c1, c2)

        if (circle1(x,y) <= 1) and (circle2(y,z) <= 1):
            X1.append(x)
            Y1.append(y)
            Z1.append(z)
            inte = inte + 1
    ans = (vol_box/float(N)) * inte
    #frac_err = abs(Fn - analyt_val)/analyt_val
    return ans



def monte_carlo_circle_area(a1,a2,b1,b2,circle,N, a = 1103515245, m = math.pow(2,32)):
    X1 = []
    Y1 = []
    vol_box = (a2 - a1) * (b2 - b1)
    ans = 0
    inte = 0

    for i in range(N):
        x = random_MLCG(a1,a2)
        y = random_MLCG(b1, b2)

        if (circle(x,y) <= 1):
            X1.append(x)
            Y1.append(y)
            inte = inte + 1
    ans = (vol_box/float(N)) * inte
    #frac_err = abs(Fn - analyt_val)/analyt_val
    return ans




def monte_carlo_integral(a1, a2, fun, N, a = 1103515245, m = math.pow(2,32)):
    inte = 0

    for i in range(N):
        #print("i", i)
        x = random_MLCG(a1, a2, a, m)
        inte = inte + fun(x)
    ans = ((a2-a1)/(N)) * inte

    return ans








def random_MLCG(L_limit, U_limit, a = 1103515245, m = math.pow(2,32), x0 = 572):
    m1 = 0

    if 'MLCG' not in globals():
        global MLCG
        MLCG = x0
    MLCG = (a * MLCG)%m
    random = MLCG/m
    random = L_limit + (U_limit - L_limit) * random

    return random






def linear_regression(X,Y, sig):
    
    a = 0
    b = 0
    chi2 = 0
    S = 0
    Sx = 0
    Sy = 0
    Sxx = 0
    Syy = 0
    Sxy = 0
    err_a = 0
    err_b = 0
    sig2_a = 0
    sig2_b = 0
    Y1 = [0 for i in range(len(X))]
    
    for i in range(len(X)):
        
        S = S + 1/(sig[i]**2)
        Sx = Sx + X[i]/(sig[i]**2)
        Sy = Sy + Y[i]/(sig[i]**2)
        Sxx = Sxx + (X[i]**2)/(sig[i]**2)
        Sxy = Sxy + (X[i]*Y[i])/(sig[i]**2)
        Syy = Syy + (Y[i]**2)/(sig[i]**2)
        
    delta = S*Sxx - (Sx**2)
    a = (Sxx*Sy - Sx*Sxy)/delta
    b = (S*Sxy - Sx*Sy)/delta
    
    for i in range(len(X)):
        Y1[i] = a + b * X[i]
        chi2 = chi2 + ((Y[i] - Y1[i])/sig[i])**2
    
    quality = chi2/(len(X)-2)
    covab = -Sx/delta
    sig2_a = Sxx/delta
    err_a = math.sqrt(sig2_a)
    sig2_b = S/delta
    err_b = math.sqrt(sig2_b)
    r2 = Sxy**2/(Sxx*Syy)
    return a,b, covab, err_a, err_b




def Chebyshev_fun(X, phi, params):
    Y = []
    for i in range(len(X)):
        sum = 0
        for j in range(len(phi)):
            sum = sum + params[j][0] * phi[j](X[i])
        Y.append(sum)
    return Y


def polynomial_fit_chebyshev(X, Y, sig, phi):
    N = len(X)
    order = len(phi)-1
    #make matrix to store A and B of AX=B
    A = make_matrix(order + 1, order + 1)
    B = make_matrix(order + 1, 1)
    #storing of A and B matrix elements
    for i in range(N):
        for j in range(order+1):
            B[j][0] = B[j][0] + ((phi[j](X[i])) * Y[i]) / (sig[i]**2)
            for k in range(order+1):
                A[j][k] = A[j][k] + (((phi[j](X[i]))*(phi[k](X[i]))) / (sig[i]**2))
    
    #print(A)
    #print(B)
    A1 = matrix_copy(A)
    A1, B = partial_pivot_inverse(A1, B)
    A1 = lu_decomposition(A1)
    B = forward_substitution(A1, B)
    B = backward_substituition(A1, B)
    chi = 0
    for i in range(N):
        sum = 0
        for j in range(order+1):
            sum = sum + B[j][0] * phi[j](i)
        chi = chi + math.pow((Y[i] - sum) / sig[i], 2)

    A_in = lu_inverse(A)

    return B, A_in



def polynomial_fit(X, Y, sig, order):
    N = len(X)
    dof = len(X)-(order+1)
    #make matrix to store A and B of AX=B
    A = make_matrix(order + 1, order + 1)
    B = make_matrix(order + 1, 1)
    #storing of A and B matrix elements
    for i in range(N):
        for j in range(order + 1):
            B[j][0] = B[j][0] + ((X[i]**j) * Y[i]) / (sig[i]**2)
            for k in range(order + 1):
                A[j][k] = A[j][k] + ((X[i]**(j+k)) / (sig[i]**2))
    
    #print(A)
    A1 = matrix_copy(A)
    A1, B = partial_pivot_inverse(A1, B)
    A1 = lu_decomposition(A1)
    B = forward_substitution(A1, B)
    B = backward_substituition(A1, B)
    chi2 = 0
    for i in range(N):
        sum = 0
        for j in range(order + 1):
            sum = sum + B[j][0] * math.pow(X[i], j)
        chi2 = chi2 + math.pow((Y[i] - sum) / sig[i], 2)

    A_in = lu_inverse(A)

    return B, A_in




def lu_inverse(A):
    I = unit_matrix(len(A))
    for j in range(len(A)):
        N = [[I[i][j]] for i in range(len(A))]
        #print('N',N)
        #print('A',A)
        A1 = matrix_copy(A)
        M, N = partial_pivot_inverse(A1, N)
        A1 = lu_decomposition(A1)
        Y = forward_substitution(A1, N)
        M = backward_substituition(A1, Y)
        for i in range(len(A1)):
            I[i][j] = M[i][0]
    return I



def lu_solution(A, B):
    A1 = matrix_copy(A)
    A1, B = partial_pivot_inverse(A1, B)
    A1 = lu_decomposition(A1)
    B = forward_substitution(A1, B)
    B = backward_substituition(A1, B)
    return B


def poly_fun(X, order, param):#return y dataset
    Y = []
    for i in range(len(X)):
        sum = 0
        for j in range(order + 1):
            sum = sum + param[j][0] * (X[i]**j)
        Y.append(sum)
    return Y




        



def jackknife(A):
    n = len(A[0])
    yj_mean = make_matrix(len(A),n) #means of data set
    yj_mean2 = make_matrix(len(A),n) #squares of means
    #print("yj_mean=",yj_mean)
    #print("yj_mean2=",yj_mean2)
    

    for i in range(len(A)):
        yj = []
        yj2 = []
        
        for k in range(len(A[0])):
            sum_y = 0.0
            for j in range(len(A[0])):
                if j != k:
                    sum_y = sum_y + A[i][j]      

            yj.append(sum_y/(n-1))
            yj2.append((sum_y/(n-1))*(sum_y/(n-1)))
       
        yj_mean[i]=yj
        yj_mean2[i]=yj2


    #print("yj_mean=",yj_mean)
    #print("yj_mean2=",yj_mean2)
        
    yjk = make_matrix(len(A),1) #mean of yj_mean
    yj2_mean = make_matrix(len(A),1) #mean of yj_mean2
    for i in range(len(A)):
        
        sum_y = 0.0
        sum_y2 = 0.0
        for j in range(n):
            sum_y = sum_y + yj_mean[i][j]
            sum_y2 = sum_y2 + yj_mean2[i][j]
        yjk[i][0] = (sum_y/n)
        yj2_mean[i][0] = (sum_y2/n)
        
    yjk2 = make_matrix(len(A),1)
    #print("yjk=",yjk)
    for i in range(len(yjk)):
        yjk2[i][0] = (yjk[i][0]*yjk[i][0])
    #print("yjk2=",yjk2)
   # print("yj2mean=",yj2_mean)
    sig_jk2 = matrix_substraction(yj2_mean,yjk2)
    
    sig = scaler_matrix_multiplication((n-1),sig_jk2)
    print("yjk",yjk)
    #print("sigjk2",sig_jk2)
    print("sig",sig)
    return yjk, sig





def diagonal_elems(A):
    diag = make_matrix(1,len(A))
    for i in range(len(A)):
        diag[0][i] = A[i][i]
    return diag

def jacobi_eigen(A, eps):
    
    
    def max_offdiag(A):
        maxd = 0.0
        for i in range(len(A)-1):
            for j in range(i+1,len(A)):
                if abs(A[i][j])>=maxd:
                    maxd = abs(A[i][j])
                    k=i
                    l=j
        return maxd,k,l
    
    def rotate(A,P,k,l):

        diff = A[l][l] - A[k][k]
        if abs(A[k][l]) < abs(diff)*1.0e-36:
            t = A[k][l]/diff
        else:
            phi = diff/(2.0*A[k][l])
            t = 1.0/(abs(phi) + math.sqrt(phi**2 + 1.0))
            if phi < 0.0:
                t = -t
        c = 1.0/math.sqrt(t**2 + 1.0)
        s = t*c
        #print("s",s)
        #print("c",c)
        tau = s/(1.0 + c)
    
        store = A[k][l]
        A[k][l] = 0.0
        A[k][k] = A[k][k] - t * store
        A[l][l] = A[l][l] + t * store
    
        for i in range(k):
            store = A[i][k]
            A[i][k] = store - s * (A[i][l] + tau * store)
            A[i][l] = A[i][i] + s * (store - tau * A[i][l])
    
        for i in range(k+1,l):
            store = A[k][i]
            A[k][i] = store - s * (A[i][l] + tau * A[k][i])
            A[i][l] = A[i][l] - s * (store - tau * A[i][l])
            
        for i in range(l+1,len(A)):
            store = A[k][i]
            A[k][i] = store - s * (A[l][i] + tau * store)
            A[l][i] = A[l][i] + s * (store - tau * A[l][i])
            
        for i in range(len(A)):
            store = P[i][k]
            P[i][k] = store - s * (P[i][l] + tau * P[i][k])
            P[i][l] = P[i][l] + s * (store - tau * P[i][l])
            
    max_rotation = 5*(len(A)**2)
    P = unit_matrix(len(A))
    for i in range(max_rotation):
        maxd, k, l = max_offdiag(A)
        if maxd < eps:
            diag = diagonal_elems(A)
            
            print("Eigenvalues = ", diagonal_elems(A))
            print("Eigenvectors =", P)
            return diag, P
        rotate(A,P,k,l)
    print("There is no convergence!")
    








def frob_norm(A):
    sum = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            sum = sum + (A[i][j]**2)
    return math.sqrt(sum)

def power_normalize(A):
    max = -1000000
    #print("d", A)
    for i in range(len(A)):
        if max <= A[i][0]:
            max = A[i][0]
    #print("max", max)
    normA = scaler_matrix_division(max,A)
    return normA


def power_method(A, x0, eps):
    i = 0
    lam0 = 1
    lam1 = 0
    while abs(lam1-lam0) >= eps:
        #print("error=",abs(lam1-lam0))
        if i != 0:
            lam0 = lam1
        
        Ax0 = matrix_multiplication(A,x0)
        AAx0 = matrix_multiplication(A,Ax0)
        #print("Ax0=",Ax0)
        #print("AAx0=",AAx0)
        dotU = inner_product(AAx0,Ax0)
        dotL = inner_product(Ax0,Ax0)
        #print("U=",dotU)
        #print("L=",dotL)
        lam1 = dotU/dotL
        
        x0 = Ax0
        i = i+1
        #print("i=",i)
        
        #print("eigenvalue=",lam1)
        ev = power_normalize(x0)
        #print ("eigenvector=",ev)
    return lam1, ev


def matrix_multiplication_on_the_fly(Afn,B):
    n = int(math.sqrt(len(B)))
    #print('B',len(B))
    #print('n',n)
    m = make_matrix(len(B),1)
    for i in range(len(B)):
        for j in range(len(B)):
            m[i][0] = m[i][0] + (Afn(i,j,n) * B[j][0])
    #print('m',m)
    return m



def conjugate_gradient_on_the_fly(Afn, B, eps):
    x0 = []
    a=[1]
    for i in range(len(B)):
        x0.append(a)
    #print('x01',x0)
    '''
    x0=make_matrix(len(B),1)
    for i in range(len(x0)):
        x0[i][0]=1
    print('B',B)
    print("x0",x0) 
    '''
    xk = matrix_copy(x0)
    

    #r0=b-Ax0
    Ax0 = matrix_multiplication_on_the_fly(Afn, x0)
    #print("Ax0",Ax0)
    rk = matrix_substraction(B, Ax0)
    #print("rk",rk)
    i = 0
    dk = matrix_copy(rk)
    #print("dk",dk)
    
    iteration=[]
    residue=[]
    while math.sqrt(inner_product(rk,rk))>=eps and i <= 1000:# and i in range(len(A)):
        adk = matrix_multiplication_on_the_fly(Afn,dk)
        #print("adk=",adk)
        rkrk = inner_product(rk, rk)
        #print("rkrk = ", rkrk)
        alpha = rkrk/inner_product(dk, adk)
        #print("alpha = ",alpha)
        xk = matrix_addition(xk, scaler_matrix_multiplication(alpha, dk))
        #print("xk1=",xk)
        rk = matrix_substraction(rk, scaler_matrix_multiplication(alpha, adk))
        #print("rk1=",rk)
        beta = inner_product(rk, rk)/rkrk
        dk = matrix_addition(rk, scaler_matrix_multiplication(beta, dk))
        
        i = i+1
        #print("norm=",math.sqrt(inner_product(rk,rk)))
        #print("i=",i)
        iteration.append(i)
        residue.append(math.sqrt(inner_product(rk,rk)))
    return xk, iteration, residue

'''
def conju_norm(A):
    sum=0
    for i in range(len(A)):
        sum = sum + abs(A[i][0])
    return sum
'''
def inner_product(A,B):

    AT = transpose(A)

    C = matrix_multiplication(AT, B)

    return C[0][0]






def conjugate_gradient(A, B, x0, eps):
    #r0 = make_matrix(len(A), 1)
    xk = matrix_copy(x0)
    
    #r0=b-Ax0
    Ax0 = matrix_multiplication(A, x0)
    #print("Ax0",Ax0)
    rk = matrix_substraction(B, Ax0)
    #print("rk",rk)
    i = 0
    dk = matrix_copy(rk)
    #print("dk",dk)
    
    iteration=[]
    residue=[]
    while math.sqrt(inner_product(rk,rk))>=eps and i <= 1000:# and i in range(len(A)):
        adk = matrix_multiplication(A,dk)
        #print("adk=",adk)
        rkrk = inner_product(rk, rk)
        #print("rkrk = ", rkrk)
        alpha = rkrk/inner_product(dk, adk)
        #print("alpha = ",alpha)
        xk = matrix_addition(xk, scaler_matrix_multiplication(alpha, dk))
        #print("xk1=",xk)
        rk = matrix_substraction(rk, scaler_matrix_multiplication(alpha, adk))
        #print("rk1=",rk)
        beta = inner_product(rk, rk)/rkrk
        dk = matrix_addition(rk, scaler_matrix_multiplication(beta, dk))
        
        #i = i+1
        #print("norm=",math.sqrt(inner_product(rk,rk)))
        #print("i=",i)
        iteration.append(i)
        residue.append(math.sqrt(inner_product(rk,rk)))
    return xk, iteration, residue



'''
def conju_norm(A):
    sum=0
    for i in range(len(A)):
        sum = sum + abs(A[i])
    return sum        
        

def conjugate_gradient(A, B, x0, eps):
    xk = x0
    rk = B- np.dot(A, x0)
    dk = rk
    i = 0
    print("norm=",conju_norm(rk))
    while i in range(len(A)):
        adk = np.dot(A, dk)
        rkrk = np.dot(rk, rk)
        
        alpha = rkrk / np.dot(dk, adk)
        print("xk0=", xk)
        xk = xk + alpha * dk
        print("xk1=", xk)
        rk = rk - alpha * adk
        if conju_norm(rk)<=eps:
            break
        else:
            beta = np.dot(rk, rk) / rkrk
            print("dk0=",dk)
            dk = rk + beta * dk
            print("dk1=",dk)
            i= i + 1
            print("i=",i)
            print("norm=",conju_norm(rk))
    return xk
    

'''

def gauss_seidel(A, B, xk0, eps):
    
    # Check: A should have zero on diagonals
    for i in range(len(A)):
        if A[i][i] == 0:
            return ("Main diagnal should not have zero!")

        
    #xk0 = make_matrix(len(A),1)
    xk1 = make_matrix(len(A),1)
    '''
    print("Guess the x matrix of length",len(A))
    for i in range(len(xk0)):
        for j in range(len(xk0[i])):
            xk0[i][j]=float(input("element:"))
    ''' 
    #list for saving no. of iteration and residue
    iteration=[]
    residue=[]


    c=0
    while inf_norm(xk1,xk0) >= eps:
        
        if c!=0:
                for i in range(len(xk1)):
                    for j in range(len(xk1[i])):
                        xk0[i][j]=xk1[i][j]
        for i in range(len(A)):
            sum1 = 0
            sum2 = 0
            for j in range(i+1,len(A[i])):
                sum2 = sum2 + (A[i][j]*xk0[j][0])
            for j in range(0,i):
                sum1 = sum1 + (A[i][j]*xk1[j][0])
            xk1[i][0] = (1/A[i][i])*(B[i][0]-sum1-sum2)
            
        c=c+1
        iteration.append(c)
        residue.append(inf_norm(xk1,xk0))
        
    return xk1, iteration, residue







def inf_norm(X,Y):
    max=0

    sum=0
    for i in range(len(X)):
        for j in range(len(X[i])):
            diff = abs(X[i][j]-Y[i][j])
            
            sum = sum + diff
            
        if sum>max:
            max = sum
    return max
        

def jacobi(A, B, xk0=None, eps=1e-4):
    #xk0 initial guess matrix
    if xk0 is None:
        xk0 = []
        a=[1]
        for i in range(len(B)):
            xk0.append(a)

    '''
    # Check: A should have zero on diagonals
    sumdiag = 0
    sumother = 0
    for i in range(len(A)):
        sumdiag = sumdiag + A[i][i]
        for j in range(len(A[i])):
            if i != j:
                sumother = sumother + A[i][j]
        if A[i][i] == 0:
            return ("Main diagnal should not have zero!")
    print("sumdiag",sumdiag)
    print("sumother",sumother)
    if sumdiag<=sumother:
        return ("Sum of diagonal must be dominant!")
    '''    
    #xk0 = make_matrix(len(A),1)
    xk1 = make_matrix(len(A),1)
    '''
    print("\nGuess the x matrix of length",len(A))
    for i in range(len(xk0)):
        for j in range(len(xk0[i])):
            xk0[i][j]=float(input("input element:"))
    '''
    #list for saving no. of iteration and residue
    iteration=[]
    residue=[]

    c=0
    while inf_norm(xk1,xk0) >= eps:
        # if c!=0, xk0=xk1
        if c!=0:
                for i in range(len(xk1)):
                    for j in range(len(xk1[i])):
                        xk0[i][j]=xk1[i][j]
        for i in range(len(A)):
            sum = 0
            for j in range(len(A[i])):
                if j!=i:
                    sum = sum + (A[i][j]*xk0[j][0])
            xk1[i][0] = (1/A[i][i])*(B[i][0]-sum)
        c=c+1
        iteration.append(c)
        residue.append(inf_norm(xk1,xk0))
    #print("c=",c)
        
    return xk1#, iteration, residue
            
                    
    
    








def partial_pivot_solution(A):
    # row loop for checking 0 on diagonal positions
    for r1 in range(len(A)-1):
        if abs(A[r1][r1]) == 0:
            # row loop for finding suitable row for interchanging
            for r2 in range(r1 + 1, len(A)):
                # row interchange
                if A[r2][r1] > A[r1][r1]:
                    a1 = A[r1]
                    A[r1] = A[r2]
                    A[r2] = a1
    return A



def partial_pivot_inverse(A, B):

    # row loop for checking 0 on diagonal positions
    for r1 in range(len(A)-1):
        if abs(A[r1][r1]) == 0:
            # row loop for finding suitable row for interchanging
            for r2 in range(r1 + 1, len(A)):
                # row interchange
                if A[r2][r1] > A[r1][r1]:
                    a1 = A[r1]
                    A[r1] = A[r2]
                    A[r2] = a1
                    b1 = B[r1]
                    B[r1] = B[r2]
                    B[r2] = b1
                    
    return A, B







def gauss_jordan_solution(A):
    #row loop
    for r1 in range(len(A)):
        #performing pivoting
        partial_pivot_solution(A)
        pivot = A[r1][r1]
        #column loop
        for c1 in range(len(A[r1])):
            A[r1][c1] = A[r1][c1]/pivot
        for r2 in range(len(A)):
            if r2 == r1 or A[r2][r1] == 0:
                pass
            else:
                factor = A[r2][r1]
                for c1 in range(len(A[r2])):
                    A[r2][c1] = A[r2][c1] - factor * A[r1][c1]

    return A


def gauss_jordan_inverse(A):
    #row loop
    if len(A) != len(A[1]):
        print("Matrix need to be square matrix")
    else:
        B = unit_matrix(len(A))
        for r1 in range(len(A)):
            # performing pivoting
            partial_pivot_inverse(A, B)
            pivot = A[r1][r1]
            #column loop
            for c1 in range(len(A[r1])):
                A[r1][c1] = A[r1][c1]/pivot
            for c2 in range(len(B[r1])):
                    B[r1][c2] = B[r1][c2] / pivot
            for r2 in range(len(A)):
                if r2 == r1 or A[r2][r1] == 0:
                    pass
                else:
                    factor = A[r2][r1]
                    for c1 in range(len(A[r2])):
                        A[r2][c1] = A[r2][c1] - factor * A[r1][c1]
                    for c2 in range(len(B[r1])):
                        B[r2][c2] = B[r2][c2] - factor * B[r1][c2]
        return B



def lu_decomposition(A):
    n = len(A)

    #perform LU Decomposition
    #Both Upper and Lower triangular matrix will be stored on A matrix together
    for j in range(n):

        # upper trianguar matrix
        for i in range(j+1):
            sum = 0
            for k in range(i):
                sum = sum + A[i][k] * A[k][j]
            #store to A matrix
            A[i][j] = A[i][j] - sum

        #lower triangular matrix
        for i in range(j+1, n):
            sum = 0
            for k in range(j):
                sum = sum + A[i][k] * A[k][j]
            # store to M matrix
            A[i][j] = (A[i][j] - sum)/A[j][j]

    return (A)



def forward_substitution(L, B):

    #creat Y matrix
    Y = [[0] for i in range(len(L))]

    # calculate Y matrix
    for i in range(len(L)):
        sum = 0
        for j in range (i+1):
            if i == j:
                pass
            else:
                sum = sum + L[i][j] * Y[j][0]
        Y[i][0] = (B[i][0] - sum)

    #return the calculated Y matrix
    return (Y)



def backward_substituition(U, Y):
    # creat x matrix
    X = [[0] for i in range(len(U))]

    # calculate x matrix
    for i in range(len(U) - 1, -1, -1):
        sum = 0
        for j in range(len(U) - 1, i - 1, -1):
            sum = sum + U[i][j] * X[j][0]
        X[i][0] = (Y[i][0] - sum) / U[i][i]

    # return the calculated x matrix
    return (X)






def transpose(A):
    #if a 1D array, convert to a 2D array = matrix
    if not isinstance(A[0],list):
        A = [A]
 
    #Get dimensions
    r = len(A)
    c = len(A[0])

    #AT is zeros matrix with transposed dimensions
    AT = make_matrix(c, r)

    #Copy values from A to it's transpose AT
    for i in range(r):
        for j in range(c):
            AT[j][i] = A[i][j]

    return AT


def scaler_matrix_multiplication(c,A):
    cA = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            cA[i][j] = c * A[i][j]
    return cA
    

def scaler_matrix_division(c,A):
    cA = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            cA[i][j] = A[i][j]/c
    return cA


def matrix_multiplication(A, B):
    AB =  [[0.0 for j in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[i])):
            add = 0
            for k in range(len(A[i])):
                multiply = (A[i][k] * B[k][j])
                add = add + multiply
            AB[i][j] = add
    return (AB)





def matrix_addition(A, B):
    
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    
    if ra != rb or ca != cb:
        raise ArithmeticError('Matrices are NOT of the same dimensions!.')
    
    C = make_matrix(ra, cb)
    
    for i in range(ra):
        for j in range(cb):
            C[i][j]=A[i][j] + B[i][j]
    return C

def matrix_substraction(A, B):
    
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    
    if ra != rb or ca != cb:
        raise ArithmeticError('Matrices are NOT of the same dimensions!.')
    
    C = make_matrix(ra, cb)
    
    for i in range(ra):
        for j in range(cb):
            C[i][j]=A[i][j] - B[i][j]
    return C



def matrix_read(M,mode):
    #read the matrix text files
    a = open(M,mode)
    A = []
    #A matrix
    for i in a:
        A.append([float(j) for j in i.split()])
    return (A)


def matrix_copy(A):
    B = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            B[i][j] = A[i][j]
    return B



def matrix_print(A):
    for i in A:
        for j in i:
            print(j, end='  ')
        print()

def unit_matrix(A):
    B = [[0 for x in range(A)] for y in range(A)]
    for i in range(len(B)):
        for j in range(len(B[i])):
            if i==j:
                B[i][j]=1
    return B


def make_matrix(N, M):
    I = [[0 for x in range(M)] for y in range(N)]
    return I