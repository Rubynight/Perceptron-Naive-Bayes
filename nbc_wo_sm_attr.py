import pandas as pd
import numpy as np
from sys import argv


class nbc():
    # constructor
    def __init__(self, trainingFile):
        self.trainingFile = trainingFile

    # Load data set
    def load(self, file_name):
        df = pd.read_csv(file_name, sep=',', quotechar='"', header=0, engine='python')
        data = df.as_matrix()
        return data

    # training
    def trainNBC(self, trainMatrix, trainCategory, attribute):
        numRows = len(trainMatrix)
        numAttributes = len(trainMatrix[0])
        # number of 1 and number of 0 in GoodForGroup vector, 1 means yes and 0 means no
        numOfYes = trainCategory.tolist().count([1])


        for i in range(numAttributes):

            # for each attribute count how many unique discrete values they have
            key, counts = np.unique(trainMatrix[:, i], return_counts=True)
            myDict = dict(zip(key, counts))
            yesProb = 0.0

            # for each discrete value of an attribute, count how many yes and no corresponds to it
            for k in myDict:
                yesCounter = 0

                for j in range(numRows):
                    # calculate numerator and by Laplace smoothing, add 1 to it
                    if trainMatrix[j][i] == k:
                        yesCounter = yesCounter + 1

                # calculate probability of P(attribute | yes) and P(attribute | no)
                # add k=possible values of attribute to the denominator
                yesProb = float(yesCounter) / numOfYes

            if i == attribute:
                return yesProb

if __name__ == '__main__':
    training_file = argv[1]

    # training
    nbc = nbc(training_file)
    data = nbc.load(training_file)
    trainMatrix = data[:, : -1]
    listClasses = data[:, -1][np.newaxis].T

    col = ['city', 'state', 'stars', 'open', 'alcohol', 'noiseLevel', 'attire',
             'priceRange', 'delivery', 'waiterService', 'smoking', 'outdoorSeating',
             'caters', 'goodForKids']

    cpd = nbc.trainNBC(trainMatrix, listClasses, col.index(argv[2]))

    print("CPD: %s = %0.4f" % (argv[2], cpd))
