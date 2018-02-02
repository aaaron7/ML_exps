# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from numpy import loadtxt, where
from numpy import * 

from pylab import scatter, show, legend, xlabel, ylabel, plot


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(X):
    return 1.0/ (1 + exp(-X))

def compute_cost(theta, X, y):
    m = X.shape[0]
    theta = reshape(theta, (len(theta) , 1))
    pass

def load_dataset():
    data = loadtxt("data1.txt", delimiter = ',')
    X = data[:, 0 : 2]
    y = data[:, 2]
    column = empty(X.shape[0])
    column.fill(1)
    X = column_stack((column, X))
    return X, y
    
def grad_ascent(data_mat,label_mat):
    data_matrix = mat(data_mat)
    label_matrix = mat(label_mat).transpose()
    # m是record的个数，而n可以当做每个record的property的数量
    m,n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 100000
        
    # 返回 n个数组组成的数组，每个数组包含1个元素， 初始化为1
    weights = ones((n,1))
    pre_cost = 0
    pre_pre_cost = 0 
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_matrix - h)
        #print(error)
        # 这里为什么是直接 x data_matrix.transpose() 而不是像公式一样 x X[j](i)
        # 核心就是本身梯度下降就是二层循环，外层是while控制总迭代，内层更新每一个weights[j]
        # 这里内层循环直接用矩阵表示，所以X[j](i)展开后累加，就是X本身
        weights = weights + alpha * data_matrix.transpose() * error
        cost = (error.getA() ** 2).sum()
        if cost > pre_cost and alpha >= 0.00001:
            alpha = alpha - 0.000005
        elif fabs(cost - pre_cost) <= 0.1 and fabs(pre_cost - pre_pre_cost) <= 0.1:
            alpha = alpha + 0.000005
        pre_pre_cost = pre_cost
        pre_cost = cost
        print("epoch %d cost is %f" % (k,cost))
        
        #break
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0,0]-weights[1,0]*x)/weights[2,0]
    print(y)
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def gen_charts(X, y, wei):
    pos = where(y == 1)
    neg = where(y == 0)
    weights = wei.getA()
    scatter(X[pos, 1], X[pos, 2], marker = 'o', c = 'b')
    scatter(X[neg, 1], X[neg, 2], marker = 'x', c = 'r')
    xlabel('F1')
    ylabel('F2')
    legend(['Fail','Pass'])
    
    # 范围+步长
    x1 = arange(30, 100, 1)
    print(weights[0,0])
    print(weights[1,0])
    print(weights[2,0])
    y1 = (-weights[0,0] - weights[1,0]*x1)/weights[2, 0]

    plot(x1,y1)
    pass
  
# exp1, MLA Test
#X,y = loadDataSet()
#weights = grad_ascent(X, y)
#plotBestFit(weights)

#
X,y = load_dataset()
weights = grad_ascent(X, y)
print(weights)
gen_charts(X, y, weights)


#data = loadtxt("data1.txt", delimiter = ',')
#
#
#X = data[:, 0 : 2]
#y = data[:, 2]
#
#print(X)
#print(y)
#
#pos = where(y == 1)
#neg = where(y == 0)
#
#scatter(X[pos, 0], X[pos, 1], marker = 'o', c = 'b')∑
#scatter(X[neg, 0], X[neg, 1], marker = 'x', c = 'r')
#
#xlabel('Feature1')
#ylabel('Feature2')
#legend(['Fail', 'Pass'])
#
#show()