import numpy as np
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from hyperopt import STATUS_OK
from plotly import graph_objects as go

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

        # initialize network 
        net = Net(dims = dims)

        # define optimizer 
        optimizer = optim.Adam(net.parameters(), lr = space['lr'])
        
        # train
        for ep in range(epoch):

            # loop over each mini batch
            for t in trainset:       
                X_temp, y_temp = t
                output = net(X_temp) # compute network output
                residual = y_temp - output
                loss = Tensor.max(quantile*residual, (quantile-1)*residual).mean() # compute the quantile loss
                optimizer.zero_grad()
                loss.backward() # back propagation
                optimizer.step() # update weights

        # predict 
        y_pred = net(Tensor(X_test_temp))
            
        # calulate validation loss and calculate its averange incrementally 
        mse_loss = nn.MSELoss()
        val_loss = mse_loss(y_pred, Tensor(y_test_temp)).detach().numpy()
        ave_val_loss = ave_val_loss + (val_loss - ave_val_loss)/counter
        counter+=1

    return {'loss':ave_val_loss, 'status': STATUS_OK}


def unpack(x):
    '''
    this is a simple helper function that allows us to fill in `np.nan` when a particular hyperparameter is not relevant to a particular trial.
    '''
    if x:
        return x[0]
    return np.nan


def plot_contour(df, z, x, y):
    '''
    this function creates a contour plot for the loss. 
    input: 
    df(dataframe): the dataframe storing the results of the hyperparameter tuning trials 
    z(string): the column name of the parameter to contour (loss)
    x(string): the column name of the x axis (hyperparameter 1)
    y(string): the column name of the y axis (hyperparameter 2)
    
    '''
    
    fig = go.Figure(
        data=go.Contour(
            z=df.loc[:, z],
            x=df.loc[:, x],
            y=df.loc[:, y],
            contours=dict(
                showlabels=True,  # show labels on contours
                labelfont=dict(size=12, color="white",),  # label font properties
            ),
            colorbar=dict(title=z, titleside="right",),
            hovertemplate=z + ": %{z}<br>max_depth: %{x}<br>" + x + ": %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title= x,
        yaxis_title= y,
        title={
            "text": x +  "vs. " + y,
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    return fig


def train(X_train, y_train, quantile, net, lr, batch_size, epoch):    
    '''
    This is the training function.
    input: 
        X_train(dataframe): the training data, X
        y_train(dataframe): the training data, y
        quantile(float): the specific quantile to be trained. values in the range of [0.0, 1.0]
        net(network): the network to train
        lr(float): learning rate
        batch_size(int): batch size
        epoch(int): epoch
    output: 
        net(network): the trained network
        net.state_dict()(dictionary): the dictionary which contains the weights and biases of the network
    '''
    # create tensor dataset
    train = TensorDataset(Tensor(X_train), Tensor(y_train))

    # create data loader from dataset
    trainset = DataLoader(train, batch_size = batch_size, shuffle = True)

    # define optimizer
    optimizer = optim.Adam(net.parameters(), lr = lr)

    # train
    for ep in range(epoch):

        # loop over each mini batch
        for t in trainset:
            X_temp, y_temp = t
            output = net(X_temp) # compute network output
            residual = y_temp - output 
            loss = Tensor.max(quantile*residual, (quantile-1)*residual).mean() # compute the quantile loss
            optimizer.zero_grad() 
            loss.backward() # back propagation
            optimizer.step() # update weights
            
    return net, net.state_dict()