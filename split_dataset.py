import random
import pandas as pd
import numpy as np
from sys import argv

X = pd.read_csv('./yelp_cat.csv', sep=',', quotechar='"', header=None, engine='python')
data = X.as_matrix()

col_name = (data[0])
data = data[1:, :]
mylist = (data.tolist())
random.shuffle(mylist)

data = np.array(mylist)
if len(argv) == 2:
    percent = argv[1]
else:
    percent = 50

train_data = data[0:int(data.shape[0] * int(percent) / 100), :]
test_data = data[int(data.shape[0] * int(percent) / 100):, :]

train_df = pd.DataFrame(data=train_data, columns=col_name)
test_df = pd.DataFrame(data=test_data, columns=col_name)

train_df.to_csv("train-set.csv", index=False, sep=',')
test_df.to_csv("test-set.csv", index=False, sep=',')
