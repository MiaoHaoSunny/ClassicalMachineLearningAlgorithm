import numpy as np
from math import log
import operator


def createDataSet():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


mydata, Labels = createDataSet()
LabelsUse = Labels.copy()
j = 0
# print(mydata, '\n', labels)


def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelcounts = {}
    for featVec in dataset:
        currentLable = featVec[-1]
        if currentLable not in labelcounts.keys():
            labelcounts[currentLable] = 0
        labelcounts[currentLable] += 1

    shannonEnt = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataset, axis, value):
    retDataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            redeuceFeatvec = featvec[:axis]
            redeuceFeatvec.extend(featvec[axis+1:])
            retDataset.append(redeuceFeatvec)
    return retDataset


def chooseBestFeatureSplit(dataset):
    numFeature = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeature):
        featList = [number[i] for number in dataset]
        uniqualVals = set(featList)
        newEntropy = 0
        for value in uniqualVals:
            subDataSet = splitDataSet(dataset, i, value)
            prob = len(subDataSet) / float(len(dataset))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 投票表决
def majorityCnt(classlist):
    classCount = {}
    for vote in classlist:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 按值将序排列
    return sortedClassCount[0][0]


def createTree(dataset, lables):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[-1]) == len(classlist):
        return classlist[-1]
    # print(classlist[0])
    # global j
    if len(classlist[0]) == 1:
        return majorityCnt(classlist)
    bestFeat = chooseBestFeatureSplit(dataset)
    bestFeatLabel = lables[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(lables[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = lables[:]
        # print(splitDataSet(dataset, bestFeat, value))
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), subLabels)
        # j += 1
    return myTree


# print(createTree(mydata, Labels))
# print(Labels)
# print(LabelsUse)
def classify(inputTree, featLables, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]  # 树的分支，子集合Dict
    featIndex = featLables.index(firstStr)  # 获取决策树第一层在featLables中的位置
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLables, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


myTree = createTree(dataset=mydata, lables=Labels)
print(classify(myTree, LabelsUse, [1, 1]))
