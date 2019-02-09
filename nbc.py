import pandas as pd
import numpy as np
from sys import argv
import math


class nbc():
    # constructor
    def __init__(self, trainingFile, testFile):
        self.trainingFile = trainingFile
        self.testFile = testFile

    # Load data set
    def load(self, file_name):
        df = pd.read_csv(file_name, sep=',', quotechar='"', header=0, engine='python')
        data = df.as_matrix()
        return data

    # training
    def trainNBC(self, trainMatrix, trainCategory):
        numRows = len(trainMatrix)
        numAttributes = len(trainMatrix[0])
        # number of 1 and number of 0 in GoodForGroup vector, 1 means yes and 0 means no
        numOfYes = trainCategory.tolist().count([1])
        numOfNo = trainCategory.tolist().count([0])
        # P(goodForGroups) = frequency of 1 appears in that column
        pGoodForGroups = numOfYes / float(numRows)

        # yes vector and no vector are the conditional probability for each attribute
        # P(attribute | yes) and P(attribute | no)

        yesList = []
        noList = []
        for i in range(numAttributes):
            yesDict = {}
            noDict = {}

            # for each attribute count how many unique discrete values they have
            key, counts = np.unique(trainMatrix[:, i], return_counts=True)
            myDict = dict(zip(key, counts))

            # for each discrete value of an attribute, count how many yes and no corresponds to it
            for k in myDict:
                yesCounter = 0
                noCounter = 0

                for j in range(numRows):
                    # calculate numerator and by Laplace smoothing, add 1 to it
                    if trainMatrix[j][i] == k and trainCategory[j] == 1:
                        yesCounter = yesCounter + 1
                    elif trainMatrix[j][i] == k and trainCategory[j] == 0:
                        noCounter = noCounter + 1

                # calculate probability of P(attribute | yes) and P(attribute | no)
                # add k=possible values of attribute to the denominator
                yesProb = float(yesCounter + 1) / (numOfYes + len(myDict))
                noProb = float(noCounter + 1) / (numOfNo + len(myDict))

                # return vectors
                yesDict[k] = math.log(yesProb)
                noDict[k] = math.log(noProb)

                # yesDict[k] = yesProb
                # noDict[k] = noProb

            yesList.append(yesDict)
            noList.append(noDict)
        return yesList, noList, pGoodForGroups

    # classify a new data set
    def classifyNBC(self, testMatrix, yesList, noList, pClass):
        numRows = len(testMatrix)
        numAttributes = len(testMatrix[0]) - 1

        # result vector
        retVect = []

        for i in range(numRows):
            probYes = 0.0
            probNo = 0.0

            for j in range(numAttributes):
                mykey = testMatrix[i][j]
                yesDict = yesList[j]
                noDict = noList[j]

                yesAverage = 0.0
                noAverage = 0.0

                for value in yesDict.values():
                    yesAverage += value

                for value in noDict.values():
                    noAverage += value

                yesAverage = yesAverage/(len(yesDict))
                noAverage = noAverage/(len(noDict))


                # calculate P(attribute | yes) for each attribute in a
                # and multiply them together as well as pClass
                if mykey in yesDict.keys():
                    probYes += yesDict.get(mykey)
                else:
                    probYes += yesAverage

                # same thing for probability of no
                if mykey in noDict.keys():
                    probNo += noDict.get(mykey)
                else:
                    probNo += noAverage

            probYes += math.log(pClass)
            probNo += math.log((1-pClass))

            # probYes += pClass
            # probNo += 1 - pClass
            #
            # sum = 0.0
            # for i in range(len(yesList)):
            #     for key in yesList[i].keys():
            #         p = yesList[i].get(key)
            #     sum += (1-p)*(1-p)
            #
            # squaredLoss = sum / len(yesList)

            # choose whichever is bigger between probability of yes and no
            # as our result for that row
            if probYes > probNo:
                retVect.append(1)
            else:
                retVect.append(0)

        return retVect


if __name__ == '__main__':
    training_file = argv[1]
    test_file = argv[2]

    # training
    nbc = nbc(training_file, test_file)
    data = nbc.load(training_file)
    trainMatrix = data[:, : -1]
    listClasses = data[:, -1][np.newaxis].T
    yesList, noList, pGoodForGroups = nbc.trainNBC(trainMatrix, listClasses)

    # classify new data set
    test = nbc.load(test_file)
    testClasses = test[:, -1][np.newaxis].T
    result = nbc.classifyNBC(test, yesList, noList, pGoodForGroups)

    # print("SQUARED LOSS=%0.4f" % squaredLoss)

    # check the zero-one loss
    sum = 0
    for a in range(len(testClasses)):
        if testClasses[a] == result[a]:
            sum = sum + 0
        else:
            sum = sum + 1

    zeroOneLoss = (float(sum) / len(testClasses))
    print("ZERO-ONE LOSS=%0.4f" % zeroOneLoss)
