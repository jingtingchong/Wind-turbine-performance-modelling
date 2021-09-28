import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable
from functools import partial
from hyperopt import fmin, hp, space_eval, tpe, STATUS_OK, Trials

from network import Net


def objective(indices, X_train, y_train, dims, epoch, quantile, space):
    '''
    This objective function performs cross validation and returns a dictionary with the average scross validation score.
    input:
    indices(generator object): generator object which contains the indices of the splitted data
    X_train(dataframe): the training data, X
    y_train(dataframe): the training data, y
    dims(Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,..., OUT_SIZE) for dimensionalities of layers
    epochs(int): the number of epochs for training
    quantile(float): the specific quantile to be trained. values in the range of [0.0, 1.0]
    space(dictionary): the search space of the hyperparameters 
    '''
    ave_val_loss = 0
    counter = 1

    # cross validation 
    for train_index, test_index in indices:
        X_train_temp, X_test_temp = X_train[train_index], X_train[test_index]
        y_train_temp, y_test_temp = y_train[train_index], y_train[test_index]
        
        # create tensor dataset
        train = TensorDataset(Tensor(X_train_temp), Tensor(y_train_temp))

        # create data loader from dataset
        trainset = DataLoader(train, batch_size = space['batch_size'], shuffle = True)

        # train
        net = Net(dims = dims)
        optimizer = optim.Adam(net.parameters(), lr = space['lr'])
        
        for ep in range(epoch):
            for t in trainset:       
                X_temp, y_temp = t
                output = net(X_temp)
                residual = y_temp - output
                loss = Tensor.max(quantile*residual, (quantile-1)*residual).mean() # quantile loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # predict 
        pred = net(Tensor(X_test_temp))
            
        # calulate validation loss and calculate its averange incrementally 
        mse_loss = nn.MSELoss()
        val_loss = mse_loss(pred, Tensor(y_test_temp)).detach().numpy()
        ave_val_loss = ave_val_loss + (val_loss - ave_val_loss)/counter
        counter+=1

    return {'loss':ave_val_loss, 'status': STATUS_OK}