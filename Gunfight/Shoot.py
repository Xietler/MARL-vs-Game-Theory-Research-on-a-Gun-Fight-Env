import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#build n*n map,input n,form an array,non-duplicate distance info,range from 0 to max
def distance(size):
    dis_list = []
    for x in range(size):
        for y in range(size):
            dis_list.append(pow(x,2)+pow(y,2))
    Arr = np.array(list(set(dis_list)))
    Arr = np.sqrt(Arr)
    Arr = np.array(sorted(Arr,reverse=True))
    return Arr

#determine the rate of bullet by a linear function,the farther,the lower rate
def function1(Arr):
    max = Arr.max()
    min = Arr.min()
    k = 1/(max-min)
    Arr = 1-k*Arr
    return Arr

#for environment to calculate the rate of linear function
def f1(dis):
    Arr = distance(10)
    max = Arr.max()
    min = Arr.min()
    k = 1/(max-min)
    acc = 1-k*dis
    return acc

#for environment to calculate the rate of Sigmoid-like function
def f2(dis):
    Arr = distance(10)
    max = Arr.max()
    min = Arr.min()
    x = (max+min)/2
    res = -1*(dis-x)
    res = 1.0/(1+1.0*np.exp(-res))
    return res

# SIGMOID = 1/1+α*e^(G(x))，alpha and G is the parameter for bullet config
def sigmoid(Arr,alpha = 1.0,G = -1):
    max = Arr.max()
    min = Arr.min()
    x = (max+min)/2
    Arr = G*(Arr-x)
    Arr = 1.0/(1+alpha*np.exp(-Arr))
    Arr[0] = 0
    Arr[len(Arr)-1]=1
    return Arr

#the payoff for only shoot env
def payoff(Arr_B,Arr_R,size):
    m = np.zeros((size,size))
    for i in range(size):
        m[i][i] = Arr_R[i] - Arr_B[i]
    for i in range(size):
        for j in range(i):
            m[i][j] = 1 - 2*Arr_B[j]
        for k in range(i+1,size):
            m[i][k] = 2*Arr_R[i] - 1
    return m

#visualize the bullet curves
def vis(min,max):
    x = np.linspace(min,max,100)
    y_f1 = function1(x)
    y_f2 = sigmoid(x)
    y_f2[0] = 1
    y_f2[len(y_f2)-1] = 0
    plt.title("线性函数和Sigmoid函数形成的不同命中率曲线")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlim((min,max))
    plt.ylim((0.0,1.0))
    plt.plot(x,y_f1,'r-',linewidth = 1,label='F(x)')
    plt.plot(x,y_f2,'b-',linewidth = 1,label='S(x)')
    plt.legend(['Linear-F(x)', 'Sigmoid-S(x)','Sigmoid-S2(x)'], loc="upper right")
    plt.savefig("./rate.png")
    plt.show()


#determine the rate of probe,jam and listener according to the distance
def calc_r(Arr,alpha = 0.2):
    return 1.0/(1+alpha*Arr)

#Blue table for inspect result
def inspectB(r):
    result = np.zeros((4, 4))
    l1 = 1 - r
    l2 = pow(1 - r, 2)
    p1 = 1 - r
    p2 = 1 - r * (1 - r)
    result[0][0] = 0
    result[0][1] = 0
    result[0][2] = r
    result[0][3] = r
    result[1][0] = r
    result[1][1] = r
    result[1][2] = 1 - p2*l1
    result[1][3] = 1 - p2*l1
    result[2][0] = r
    result[2][1] = r
    result[2][2] = 1 - p1 * l1
    result[2][3] = 1 - p1 * l1
    result[3][0] = 1- l2
    result[3][1] = 1 - l2
    result[3][2] = 1 - p2 * l2
    result[3][3] = 1 - p2 * l2
    return result

#red table for inspect result
def inspectR(r):
    result = np.zeros((4,4))
    l1 = 1 - r
    l2 = pow(1-r,2)
    p1 = 1 - r
    p2 = 1 - r *(1-r)
    result[0][0] = 0
    result[0][1] = r
    result[0][2] = r
    result[0][3] = 1 - l2
    result[1][0] = 0
    result[1][1] = r
    result[1][2] = r
    result[1][3] = 1 - l2
    result[2][0] = r
    result[2][1] = 1 - p2*l1
    result[2][2] = 1 - p1*l1
    result[2][3] = 1 - p2*l2
    result[3][0] = r
    result[3][1] = 1 - p2*l1
    result[3][2] = 1 - p1*l1
    result[3][3] = 1 - p2*l2
    return result

#test the payoff by change the parameters
def payoff2(Arr_R,Arr_B):
    Arr = distance(10)
    r = calc_r(Arr)
    result = np.zeros((204,204))
    for i in range(51):
        for j in range(51):
            if i == j:
                insB = inspectB(r[i])
                insR = inspectR(r[i])
                for x in range(4):
                    for y in range(4):
                        result[4*i+x][4*j+y] = insR[x][y]*Arr_R[i] - insB[x][y]*Arr_B[i]
            elif i > j:
                insB = inspectB(r[j])
                for x in range(4):
                    for y in range(4):
                        result[4*i+x][4*j+y] = 1 - 2*insB[x][y]*Arr_B[j]

            elif i < j:
                insR = inspectR(r[i])
                for x in range(4):
                    for y in range(4):
                        result[4*i+x][4*j+y] = 2*insR[x][y]*Arr_R[i] - 1
            else:
                pass
    return result


if __name__ == '__main__':

    # Arr = distance(10)
    # func1_Arr = function1(Arr)
    # sig = sigmoid(Arr)
    # m = payoff(func1_Arr,sig,len(sig))
    # np.set_printoptions(precision=4,suppress=True)
    # dataf = pd.DataFrame(m)
    # dataf.to_csv('Payoff2.csv',float_format='%.4f',)
    # print(m)
    # vis(Arr.min(),Arr.max())
    Arr = distance(10)
    func1_Arr = function1(Arr)
    sig = sigmoid(Arr)
    m = payoff2(func1_Arr,sig)
    np.set_printoptions(precision=4,suppress=True)
    dataf = pd.DataFrame(m)
    dataf.to_csv('./payoff/PayoffPJL.csv',float_format='%.4f')
