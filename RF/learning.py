import numpy as np
import pandas as pd
from functools import partial
from hyperopt import fmin, hp, space_eval, tpe, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from plotly import graph_objects as go
from sklearn.model_selection import KFold

def objective(skf, X_train, y_train, space):
    '''
    This objective function performs cross validation and returns a dictionary with the average scross validation score.
    input:
        skf(sklearn kfold): kfold cross validator 
        X_train(dataframe): the training data, X
        y_train(dataframe): the training data, y
        space(dictionary): the search space of the hyperparameters 
    '''
    # define model
    model = RandomForestRegressor(criterion = 'mse', max_depth = space['max_depth'],
                                bootstrap = True, 
                                n_estimators = space['n_estimators'], 
                                max_samples = space['max_samples'],
                                n_jobs = -1, verbose = 2
                                )

    ave_val_loss = 0
    counter = 1

    # split data
    split_indices = skf.split(X_train, y_train) 

    # cross validation 
    for train_index, test_index in split_indices:
        X_train_temp, X_test_temp = X_train[train_index], X_train[test_index]
        y_train_temp, y_test_temp = y_train[train_index], y_train[test_index]
        
        # fit
        model.fit(X_train_temp, y_train_temp.flatten()) 

        # predict
        y_pred = model.predict(X_test_temp) 
    
        # calulate validation loss and calculate its averange incrementally 
        val_loss = metrics.mean_squared_error(y_test_temp, y_pred)
        # update average prediction loss incrementally 
        ave_val_loss = ave_val_loss + (val_loss - ave_val_loss)/counter
        counter+=1

    return {'loss': ave_val_loss, 'status': STATUS_OK }


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