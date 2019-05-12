from math import *
#Question 3#
def bin_coef(k,n):
    return factorial(n)/(factorial(k)*factorial(n-k))

def f_test(x):
    if (x-90 > 0):
        return x-90
    else:
        return 0

def price1(N,r_N,h_N,b_N,s,f):
    q_N=(r_N-b_N)/(h_N-b_N)
    res=1/(1+r_N)**N
    sum=0
    for k in range(0,N):
        sum+=f(s*(1+h_N)**k*(1+b_N)**(N-k))*bin_coef(k,N)*q_N**k*(1-q_N)**(N-k)
    return (res * sum)
#Question 4#
price1(30,0.01,0.05,-0.05,100,f_test)
#Question 5#

def pricer2bis(N,r_N,h_N,b_N,s,f,k):
    if (k==N):
        return f_test(s)
    else :
        q_N = (r_N-b_N)/(h_N-b_N)
        esp = pricer2bis(N,r_N,h_N,b_N,s*(1+h_N),f_test,k+1)*q_N + pricer2bis(N,r_N,h_N,b_N,s*(1+b_N),f_test,k+1)*(1-q_N)
        return esp / (1+r_N)

#Question 6#
pricer2bis(3,0.01,0.05,-0.05,100,f_test,0)

def pricer2bisprint(N,r_N,h_N,b_N,s,f,k):
    if (k==N):
        return f_test(s)
    else :
        q_N = (r_N-b_N)/(h_N-b_N)
        noeud1 = pricer2bisprint(N,r_N,h_N,b_N,s*(1+h_N),f_test,k+1)
        noeud2 = pricer2bisprint(N,r_N,h_N,b_N,s*(1+b_N),f_test,k+1)
        print("v",k+1,"(1+h_N)",noeud1)
        print("v",k+1,"(1+b_N)",noeud2)
        if (k==0):
            print("v",0,noeud1*q_N / (1+r_N) + noeud2*(1-q_N) / (1+r_N))
        return (noeud1*q_N / (1+r_N) + noeud2*(1-q_N) / (1+r_N))

pricer2bisprint(3,0.01,0.05,-0.05,100,f_test,0)   
'''
v 3 (1+h_N) 25.762500000000003
v 3 (1+b_N) 14.737499999999997
v 3 (1+h_N) 14.737500000000011
v 3 (1+b_N) 4.762499999999989
v 2 (1+h_N) 21.14108910891089
v 2 (1+b_N) 10.641089108910894
v 3 (1+h_N) 14.737500000000011
v 3 (1+b_N) 4.762499999999989
v 3 (1+h_N) 4.762500000000003
v 3 (1+b_N) 0
v 2 (1+h_N) 10.641089108910894
v 2 (1+b_N) 2.829207920792081
v 1 (1+h_N) 16.773355553377122
v 1 (1+b_N) 7.441917459072641
v 0 12.911663678866663
'''
#Question 7#
import matplotlib.pyplot as plt     
T=[k for k in range(5,16)]
Y1=[]
Y2=[]
for e in T:
    Y1.append(price1(e,0.01,0.05,-0.05,100,f_test))
    Y2.append(pricer2bis(e,0.01,0.05,-0.05,100,f_test,0))
plot1, = plt.plot(T,Y1,label='pricer1',color='red')
plot2, = plt.plot(T,Y2,label='pricer2',color='blue')
plt.legend([plot1,plot2],['pricer1','pricer2'])
plt.title('Comparaison des deux pricers')
plt.xlabel('N')
plt.ylabel('Prix option')
plt.show()

#Question 10#
def fq10(x):
    return max(x-100,0)
'''nbre d'actifs risqués achetés en 0'''
alpha0=(pricer2bis(2,0.03,0.05,-0.05,1.05*100,fq10,1) - pricer2bis(2,0.03,0.05,-0.05,0.95*100,fq10,1))/((0.05+0.05) * 100) 
print(alpha0)
'''nbre d'actifs sans risque achetés en 0'''                     
beta0=(pricer2bis(2,0.03,0.05,-0.05,0.95*100,fq10,1)*1.05 - pricer2bis(2,0.03,0.05,-0.05,1.05*100,fq10,1)*0.95)/((0.05+0.05) * (1.03)**1)   
print(beta0)             
'''nbre d'actifs risqués achetés en 1'''        
alpha1=(fq10(1.05*100) - fq10(0.95*100))/((0.05+0.05) * 100) 
print(alpha1)      
'''nbre d'actifs sans risque achetés en 1'''                
beta1=(fq10(0.95*100)*1.05 - fq10(1.05*100)*0.95)/((0.05+0.05) * (1.03)**2)  
print(beta1)                       
alpha1*1.05*100+beta1*1.03**2   '''5.000000000000007'''
fq10(1.05*100)                  '''5.0''' 
''' A la date terminale, le portefeille vérifie bien la relation'''

#Quesion 12#
def fq12(x):
    return max(100-x,0)
import numpy as np
def price3(n,s,r,sigma,T,f):
    sum=0
    for i in range(1,n):
        sum+=exp(-r*T)*f(s*exp((r-sigma**2/2)*T+sigma*sqrt(T)*np.random.normal()))
    return sum/n
#Question 13#
T_n=[10**5*k for k in range(1,11)]
Y1=[]
for e in T_n:
    Y1.append(price3(e,100,0.01,0.1,1,fq12))
plot1, = plt.plot(T_n,Y1)
plt.legend([plot1],['price3'])
plt.title('Convergence estimateur de p')
plt.xlabel('N')
plt.ylabel('Prix option')
plt.show()
#Question 14#
#Question 15#
from scipy.stats import norm
def put(s,r,sigma,T,K):
    d=(log(s/K)+(r+sigma**2/2)*T)/(sigma*sqrt(T))
    p = -s * norm.cdf(-d) + K*exp(-r*T)*norm.cdf(-d+sigma*sqrt(T))
    return p
#Question 16#
put(100,0.04,0.1,1,100)
#Question 17#
T_n=[10**5*k for k in range(1,11)]
Y1=[]
Y2=[]
for e in T_n:
    Y1.append(price3(e,100,0.01,0.1,1,fq12))
    Y2.append(put(100,0.01,0.1,1,100))
plot1, = plt.plot(T_n,Y1)
plot2, = plt.plot(T_n,Y2)
plt.legend([plot1,plot2],['price3','put'])
plt.title('Comparaison estimateur de p et put')
plt.xlabel('N')
plt.ylabel('Prix option')
plt.show()
#Question 18#
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
T_s =[20 * k for k in range(1,11)]
T_T =[1/12,1/6,1/4,1/3,1/2,1]
x=np.array(T_s)
y=np.array(T_T)
X, Y = np.meshgrid(x, y)
put2 = np.vectorize(put)
zs = np.array(put2(np.ravel(X),0.01, 0.1,np.ravel(Y),100))
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('s variation')
ax.set_ylabel('T variation')
ax.set_zlabel('prix option payant max(K-S_t,0)')

plt.show()

#Question 19#
X=[10 * k for k in range(1,101)]
Y1=[]
Y2=[]
for elem_x in X:
    r_N19=0.04*1/elem_x
    h_N19=r_N19+0.2*sqrt(1)/sqrt(elem_x)
    b_N19=r_N19-0.2*sqrt(1)/sqrt(elem_x)
    Y1.append(price1(elem_x,r_N19,h_N19,b_N19,100,fq12))
    Y2.append(put(100,0.04,0.2,1,100))
plt.plot(X,Y1)
plt.plot(X,Y2)
plt.show()

#Question 20#









