import my_library as ml
import math
import matplotlib.pyplot as plt
import numpy as np

delt = 0.008
delx = 0.1
L = 2
n = 20+1
steps = [0,10,20,50,100,200,500]
time = [delt*steps[i] for i in range(len(steps))]



#define u0 matrix(initial conditions)
u0 = ml.make_matrix(n,1)
#print((u0))
u0[0][0] = 0
u0[-1][0] = 0
x = 0
i=0
while x*delx<=2:
    u0[x][0] = 20 * abs(math.sin(math.pi*x*delx))
    x = x + 1
#print(u0)


x = np.arange(0,L+delx,delx)
for i in range(len(time)):
    u1 = ml.pde_explicit(u0,n,delx,delt,time[i])
    u1 = ml.transpose(u1)
    u1 = u1[0]
    plt.plot(x,u1,label='TimeSteps = %s'%steps[i])

plt.legend()
plt.grid()
plt.xlabel('lengths(unit)')
plt.ylabel('temperature(degree celcius)')

plt.savefig('plot_q3.png')
plt.show()

'''
As the time steps increases initial the heat distribtion starts diffuses more and more
and the two peaks comes down and the the minimia between two peaks starts rising.
After significant time steps (200 and above) due to diffusion,on the center portions of 
the rod the temperature become maximum and as timesteps increases the entire curve
(along with magnitude of center parts) decreases. 

'''