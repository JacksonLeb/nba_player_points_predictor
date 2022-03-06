from base64 import encode
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''

# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

class NBADataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = pd.read_csv('2021-2022 NBA Player Stats.csv', delimiter=';', dtype=None, encoding = "ISO-8859-1")
        print(xy.shape)
        print(type(xy))
        self.n_samples = xy.shape[0]
        x = xy.iloc[:, 5:].values
        print(x)

        y = xy.iloc[:, [2]].values
        y = pd.DataFrame(y)
        print(y.shape)
        #finding position and frequencey for dummy values
        print(y[0].value_counts())
        #applying cat codes
        y[0] = y[0].astype('category')
        print(y)
        print("DATA TYPE")
        print(y[0].dtypes)
        cat_column = y.select_dtypes(['category']).columns  
        y[cat_column] = y[cat_column].apply(lambda x: x.cat.codes)
        print(y)
        print(y.shape)

        # here the third column is the class label, the rest are the features
        self.x_data = torch.tensor(x) # size [n_samples, n_features]
        self.y_data = torch.tensor(y[0], dtype=torch.int) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# create dataset
dataset = NBADataset()
x_data = torch.tensor(dataset.x_data, dtype=torch.float16)
print(type(x_data[15][15]))
print(type(dataset.y_data[0][15]))

X_train, X_test, y_train, y_test = train_test_split(x_data, dataset.y_data, test_size=0.2, random_state=1234)
# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
print(features, labels)

#model intialization
model = Model(features)

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0)

# convert to an iterator and look at one random sample
dataiter = iter(train_loader)
data = dataiter.next()
features, labels = data
print(features, labels)

# loss and optimization 
num_epochs = 100
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#training loop
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#accuracy
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')
