
#def train():
from __future__ import division
import numpy as np

import torch.utils.data as utils_data
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from SparseLatticeNetwork import SplatNet

def make_one_hot(targets, C):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1) 
    return one_hot.reshape(targets.shape[1],targets.shape[2],C)
  
def accuracy(y, labels):
  return ((y == labels).float()).mean()

X_train = np.load('points_train.npy')
y_train = np.load('labels_train.npy')
X_val = np.load('points_val.npy')
y_val = np.load('labels_val.npy')

num_classes = np.unique(raw_y).shape[0] #torch.from_numpy(np.unique(raw_y).shape[0]) 
X_train = Variable(torch.from_numpy(X_train).float())
y_train = Variable(torch.tensor(torch.from_numpy(y_train).float(), dtype=torch.long)).cuda()
X_val = Variable(torch.from_numpy(X_val).float())
y_val = Variable(torch.tensor(torch.from_numpy(y_val).float(), dtype=torch.long)).cuda()

print("Number of classes: ", num_classes)
print(X.shape, y.shape, num_classes)

X_train = Variable(X_train)
y_train = Variable(y_train)
y_one_hot_train = make_one_hot(y_train.reshape(1,X_train.shape[0], X_train.shape[1]), num_classes)
y_one_hot_train = Variable(y_one_hot_train, requires_grad=True)
print("Training data:", X_train.shape, y_train.shape, y_one_hot_train.shape)

X_val = Variable(X_val)
y_val = Variable(y_val)
y_one_hot_test = make_one_hot(y_val.reshape(1,X_val.shape[0], X_val.shape[1]), num_classes)
y_one_hot_test = Variable(y_one_hot_test, requires_grad=True)
print("Testing data:", X_val.shape, y_val.shape, y_one_hot_test.shape)

y_one_hot = make_one_hot(y.reshape(1,X.shape[0], X.shape[1]), num_classes)
y_one_hot = Variable(y_one_hot, requires_grad=True)
print("Data:", y_one_hot.shape, X.shape)

model = SplatNet(num_classes)

if(torch.cuda.is_available()):
  print('Cuda is available')
  model = model.cuda()
    
criterion = nn.CrossEntropyLoss()
count        = 0
learningRate = 0.001 
numEpochs    = 2 
weightDecay  = 0.0001
momentum     = 0.9
batch_size   = 32
optimizer    = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
#optimizer    = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum, weight_decay=weightDecay)
verbose = False

print("Training Starting..")
for epoch in range(numEpochs):
    train_loss  = 0
    train_accuracy = []
    val_accuracy =[]
    for i in range(X_train.shape[0]):
        optimizer.zero_grad()
        out = model(X[i]).squeeze()
        _, yi = torch.max(torch.transpose(out,dim0=0, dim1=1),dim=1)
        yi = yi.cuda()
        loss = criterion(y_one_hot_train[i], yi)
        loss.backward()
        optimizer.step()
        train_loss += loss
        train_accuracy.append(accuracy(y_train[i], yi))

    
    
    for i in range(X_val.shape[0]):
      output_val = model(X_val[i]).squeeze()
      _, yi = torch.max(torch.transpose(output_val,dim0=0, dim1=1),dim=1)
      yi = yi.cuda()
      val_loss = criterion(y_one_hot_test[i], yi)
      val_acc = accuracy(y_val[i], yi)
      val_accuracy.append(val_acc)
     
    train_accuracy = torch.mean(torch.tensor(train_accuracy))
    val_accuracy = torch.mean(torch.tensor(val_accuracy))
    # report scores per epoch
    print('Epoch [%d/%d], Train_Acc: %.2f, Train_loss: %.2f, Val_Acc: %.2f, Val_Loss: %.2f',(epoch+1, numEpochs, train_loss/(i+1), train_accuracy, val_loss/(i+1), val_accuracy))
