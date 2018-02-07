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
    # m是record的个数，假设feature的数量为d，则此处的n=d+1, 因为会多一列全为1的列，用于当做x0
    m,n = shape(data_matrix)


    #feature scaling
    max_array =  amax(data_matrix, axis=0)
    min_array = amin(data_matrix, axis=0)
    mean_array = mean(data_matrix,axis=0)
    print max_array[0,1], max_array[0,0], max_array[0,2]
    for i in xrange(0,m):
        for j in xrange(1,n):
            data_matrix[i,j] = (data_matrix[i, j] - mean_array[0, j]) / (max_array[0,j] - min_array[0, j])
    alpha = 0.001
    max_cycles = 3000
        
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
    
        # 根据每次cost的增加来动态的调整learning rate
        cost = (error.getA() ** 2).sum()
        alpha = updateAlpha(alpha, cost,pre_cost, pre_pre_cost)
        pre_pre_cost = pre_cost
        pre_cost = cost
        print("epoch %d cost is %f" % (k,cost))
    
    return weights, max_array, min_array, mean_array

def updateAlpha(alpha, cost, pre_cost, pre_pre_cost):
    if cost > pre_cost and alpha >= 0.00001:
        alpha = alpha - 0.000005
    elif fabs(cost - pre_cost) <= 0.1 and fabs(pre_cost - pre_pre_cost) <= 0.1:
        alpha = alpha + 0.000005
    return alpha

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

def gen_charts(X, y, wei,max_a, min_a, mean_a):
    pos = where(y == 1)
    neg = where(y == 0)
    weights = wei.getA()
    scatter(X[pos, 1], X[pos, 2], marker = 'o', c = 'b')
    scatter(X[neg, 1], X[neg, 2], marker = 'x', c = 'r')
    xlabel('F1')
    ylabel('F2')
    legend(['Fail','Pass'])
    
    # 范围+步长
    x1 = arange(30, 100, 0.5)
    originalX1 = copy(x1)
    print(weights[0,0])
    print(weights[1,0])
    print(weights[2,0])
    x1 = (x1 - mean_a[0, 1]) / (max_a[0, 1] - min_a[0,1])
    print x1

    y1 = (-weights[0,0] - weights[1,0]*x1)/weights[2, 0]
    y1 = y1 * (max_a[0,2] - min_a[0, 2]) + mean_a[0,2]
    import matplotlib.pyplot as plt

    plot(originalX1,y1)
    plt.show()
    pass
  
# exp1, MLA Test
# X,y = loadDataSet()
# weights = grad_ascent(X, y)
# plotBestFit(weights)

# exp2 second data.txt
X,y = load_dataset()
originalX = copy(X)
weights,max_a, min_a, mean_a = grad_ascent(X, y)
print weights
gen_charts(originalX, y, weights, max_a, min_a, mean_a)


