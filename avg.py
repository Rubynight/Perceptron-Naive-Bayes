import pandas as pd
from sys import argv


class vanilla():
    # constructor
    def __init__(self, trainingFile, testFile):
        self.trainingFile = trainingFile
        self.testFile = testFile

    # Load data set
    def load(self, file_train, file_test):
        df_train = pd.read_csv(file_train, sep=',', quotechar='"', header=0, engine='python')
        df_test = pd.read_csv(file_test, sep=',', quotechar='"', header=0, engine='python')

        split_point = df_train.shape[0]

        frames = [df_train, df_test]
        df = pd.concat(frames)
        feature = df.iloc[:, :-1]
        label = df.iloc[:, -1]
        feature = pd.get_dummies(feature, columns=feature.columns.values)

        frames = [feature, label]
        df = pd.concat(frames, axis=1)

        df_train = df.iloc[0:split_point, :]
        df_test = df.iloc[split_point:-1, :]
        return df_train.as_matrix(), df_test.as_matrix()

    # training
    def trainVanilla(self, trainMatrix, MaxIter):
        # initialize weights and bias
        weights = []
        for i in range(len(trainMatrix[0])-1):
            weights.append(0)
        bias = 0.0

        # initialize cached weights and bias
        cachedWeights = []
        for i in range(len(trainMatrix[0])-1):
            cachedWeights.append(0)
        cachedBias = 0.0

        counter = 1
        for iter in range(MaxIter):
            for row in trainMatrix:
                prediction = vanilla.predict(row, weights, bias)
                error = row[-1] - prediction
                if error:
                    # update bias
                    bias = bias + error
                    # update weights
                    for i in range(len(row) - 1):
                        weights[i] = weights[i] + error * row[i]
                    # update cached bias
                    cachedBias = cachedBias + error * counter
                    # update cached Weights
                    for i in range(len(row) - 1):
                        cachedWeights[i] = cachedWeights[i] + error * row[i] * counter
                counter = counter + 1
        # return averaged bias and weight
        bias = bias - cachedBias/counter
        for i in range(len(weights)):
            weights[i] = weights[i] - cachedWeights[i]/counter
        return bias, weights

    # Make a prediction with weights
    def predict(self, row, weights, b):
        activation = 0.0
        for i in range(len(row) - 1):
            activation += weights[i] * row[i]

        activation += b

        if activation >= 0.0:
            return 1
        else:
            return 0


if __name__ == '__main__':
    training_file = argv[1]
    test_file = argv[2]

    if len(argv) == 4:
        MaxIter = argv[3]
    else:
        MaxIter = 2

    # train
    vanilla = vanilla(training_file, test_file)
    train, test = vanilla.load(training_file, test_file)
    bias, weights = vanilla.trainVanilla(train, int(MaxIter))
    # print(bias, weights)

    # predict
    predict = []
    for row in test:
        result = vanilla.predict(row, weights, bias)
        predict.append(result)
    # print(predict)

    # class labels
    labels = []
    for row in test:
        labels.append(row[-1])
    # print(labels)

    # check the zero-one loss
    sum = 0
    for a in range(len(test)):
        if labels[a] == predict[a]:
            sum += 0
        else:
            sum += 1
    zeroOneLoss = (float(sum) / len(test))
    print("ZERO-ONE LOSS=%0.4f" % zeroOneLoss)