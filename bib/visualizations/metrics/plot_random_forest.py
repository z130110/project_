import numpy as np
import sys
sys.path.append("/home/Workdir/bib/constants")
from accuracy import PlotCustomAccuracy

import directories as DIR

_dir = DIR.JSON_RESULTS + "data10/"



test_val_loss = np.load(_dir + "new_test_val_loss.npy")
train_loss = np.load(_dir + "new_train_loss.npy")
train_val_loss = np.load(_dir + "new_train_val_loss.npy")

print(_dir)


print(test_val_loss.shape)
print(train_loss.shape)
print(train_val_loss.shape)

PlotCustomAccuracy(train_loss,train_val_loss, test_val_loss,"data10").plot()
