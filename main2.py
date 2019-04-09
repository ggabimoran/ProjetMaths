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
def pricer2(N,r_N,h_N,b_N,s,f):
    if (N == 0) :
        return f(s)
    else :
        q_N = (r_N-b_N)/(h_N-b_N)
        return (pricer2(N-1,r_N,h_N,b_N,s,f) + (1+h_N)*q_N + (1+b_N)*(1-q_N)) / (1+r_N)

def pricer2_print(N,r_N,h_N,b_N,s,f):
    if (N == 0) :
        return f(s)
    else :
        q_N = (r_N-b_N)/(h_N-b_N)
        return (pricer2(N-1,r_N,h_N,b_N,s,f) + (1+h_N)*q_N + (1+b_N)*(1-q_N)) / (1+r_N)


def f(x):
    return max(x-90,0)
#Question 6#
for i in range(0,31):
    print("v",i,":",pricer2(i,0.01,0.05,-0.05,100,f))        
#Question 7#
import matplotlib.pyplot as plt     
T=[k for k in range(5,16)]
Y1=[]
Y2=[]
for e in T:
    Y1.append(price1(e,0.01,0.05,-0.05,100,f))
    Y2.append(pricer2(e,0.01,0.05,-0.05,100,f))
plt.plot(T,Y1,label='pricer1',color='red')
plt.plot(T,Y2,label='pricer2',color='blue')
plt.legend([Y1,Y2],['pricer1','pricer2'])
plt.title('Comparaison des deux pricers')
plt.xlabel('N')
plt.ylabel('Prix')
plt.show()

#Question 8#











