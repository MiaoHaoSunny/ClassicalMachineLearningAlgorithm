"""
author:缪浩
created:2018-12-11
purpose:实现k-means算法
"""

import numpy as np
# from numpy import *
import matplotlib.pyplot as plt
import pandas as pd

# 使用ris鸢尾花数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pd.read_csv(url, names=names)

# 将鸢尾花类别用0, 1, 2表示
# print(dataset)
dataset['class'][dataset['class'] == 'Iris-setosa'] = 0
dataset['class'][dataset['class'] == 'Iris-versicolor'] = 1
dataset['class'][dataset['class'] == 'Iris-virginica'] = 2
# print(dataset)


# 计算欧氏距离
def DistCalculation(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


# 随机选取中心点
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = np.float(np.max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.randn(k, 1))
    return centroids


# print(shape(dataset)[1])
def randChosenCent(dataSet, k):
    m = np.shape(dataSet)[0]
    centroidsIndex = []
    dataIndex = list(range(m))
    for i in range(k):
        randIndex = np.random.randint(0, len(dataIndex))
        centroidsIndex.append(dataIndex[randIndex])
        del dataIndex[randIndex]

    centroids = dataSet.iloc[centroidsIndex]
    return np.mat(centroids)


def kMeans(dataSet, k):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))

    centroids = randChosenCent(dataSet, k)
    print('Original centroids: ', centroids)

    clusterChanged = True

    iterTime = 0

    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1

            for j in range(k):
                distJI = DistCalculation(centroids[j, :], dataSet.values[i, :])

                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        iterTime += 1
        sse = np.sum(clusterAssment[:, 1])
        print('The SSE of %d' % iterTime + 'th iteration is %f' % sse)

        for cent in range(k):
            ptsInclust = dataSet.iloc[np.nonzero(clusterAssment[:, 0].A == cent)[0]]

            centroids[cent, :] = np.mean(ptsInclust, axis=0)
    return centroids, clusterAssment


def kMeansSSE(dataSet, k, distMeas=DistCalculation, createCent=randChosenCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    print('Initial centroids: ', centroids)
    sseOld = 0
    sseNew = np.inf
    iterTime = 0
    while abs(sseNew - sseOld) > 0.0001:
        sseOld = sseNew
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet.values[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            clusterAssment[i, :] = minIndex, minDist**2

        iterTime += 1
        sseNew = sum(clusterAssment[:, 1])
        print('The SSE of %d' % iterTime + 'th iteration is %f' % sseNew)

        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]

            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


# 数据显示
def datashow(dataSet, k, centroids, clusterAssment):
    num, dim = np.shape(dataSet)

    if dim != 2:
        print('Your dimension is not 2!!!!!')
        return 1
    marksamples = ['or', 'ob', 'og', 'ok', '^r', '^b', '<g']
    if k > len(marksamples):
        print('Your k is so large, please add length of the marksample!!!')
        return 1

    for i in range(num):
        markIndex = int(clusterAssment[i, 0])

        plt.plot(dataSet.iat[i, 0], dataSet.iat[i, 1], marksamples[markIndex], markersize=6)
    markcentroids = ['o', '*', '^']
    label = ['0', '1', '2']
    c = ['yellow', 'pink', 'red']

    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], markcentroids[i], markersize=15, label=label[i], c=c[i])
        plt.legend(loc='upper left')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')

    plt.title('K-means cluster result')
    plt.show()


def targetshow(dataSet, k, labels):
    num, dim = np.shape(dataSet)
    label = ['0', '1', '2']
    marksamples = ['ob', 'or', 'og', 'ok', '^r', '^b', '<g']
    # 通过循环的方式，完成分组散点图的绘制
    for i in range(num):
        plt.plot(dataSet.iat[i, 0], dataSet.iat[i, 1], marksamples[int(labels.iat[i, 0])], markersize=6)
    for i in range(0, num, 50):
        plt.plot(dataSet.iat[i, 0], dataSet.iat[i, 1], marksamples[int(labels.iat[i, 0])], markersize=6,
                 label=label[int(labels.iat[i, 0])])
    plt.legend(loc='upper left')
    # 添加轴标签和标题

    plt.xlabel('sepal length')
    plt.ylabel('sepal width')

    plt.title('iris true result')  # 标题

    # 显示图形
    plt.show()


def originalDataShow(dataSet):
    num, dim = np.shape(dataSet)
    marksample = ['ob']
    for i in range(num):
        plt.plot(dataSet.iat[i, 0], dataSet.iat[i, 1], marksample[0], markersize=5)
    plt.title('Original DataSet')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.show()


datamat = dataset.loc[:, ['sepal_length', 'sepal_width']]

labels = dataset.loc[:, ['class']]

originalDataShow(datamat)

k = 3
mycentroids, clussterAssment = kMeans(datamat, k)
datashow(datamat, k, mycentroids, clussterAssment)
targetshow(datamat, 3, labels)
