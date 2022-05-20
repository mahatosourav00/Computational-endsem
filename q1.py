import my_library as ml
import math
import matplotlib.pyplot as plt

#no. of walks
M = 500
#no. of steps
N = 200

#global MLCG
#MLCG = 572

X1, Y1, r_rms1, avg_x1, avg_y1, rad_dis1 = ml.random_walk(M, N)





print("Rrms = ", r_rms1)
print("rootN = ",math.sqrt(N))
#print("Average X = ", avg_x1)
#print("Average Y = ", avg_y1)
#print("Radial distance R = ", rad_dis1)

for i in range(5):
    plt.plot(X1[i],Y1[i])
    plt.title('First 5 Random walks for steps,N = 200')
    plt.grid()
plt.savefig('plot_q1.png')
plt.show()
'''

Rrms =  14.068907098196034
rootN =  14.142135623730951

'''