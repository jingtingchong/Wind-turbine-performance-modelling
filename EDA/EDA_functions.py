import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Note: The functions in this file only applies to dataframes containing the columns as listed below: 
## 'ts'(datetime): pandas datetime in the format '%d-%b-%Y %H:%M:%S'
## 'instanceID'(string): turbine ID
## 'Wind_speed' (float): wind speed
## 'TI'(float): turbulence intensity
## 'Temperature'(float): Ambient temperature
## 'Power' (float): measured turbine power output 
## 'value'(boolean): FullPerformanceFlag boolean value. this column is only required for the function plot_fullperformanceflag

def plot_timeseries(df, var):
    '''
    this function plots the time series 
    input: 
        df(dataframe): the dataframe to plot
        var(string): the name of the variable to plot
    output: 
        none
    '''
    fig, ax = plt.subplots(figsize = (16,3))
    ax.plot(df['ts'], df[var])
    ax.set_ylabel(var)
    plt.show() 


def plot_fullperformanceflag(df, order, ticklabels, figsize = (16,4), orient = "v"):
    '''
    this function plots the bar chart of full performance flag vs sample count for each turbine 
    input: 
        df(dataframe): the dataframe to plot
        order(list): the list of the turbine IDs in order 
        ticklabels(list): the list of the turbine IDs name, to be displayed in the plot
        figsize(tuple): the figure size 
        orient(string): "v" for vertical orientation, "h" for horizontal orientation
    output: 
        none
    '''
    fig, ax = plt.subplots(figsize = figsize)

    if orient == "v": 
        ax = sns.barplot(y="ts", x="instanceID", hue = "value", data=df, orient = orient, order = order)
        plt.ylabel("Sample count")
        plt.xlabel("AWI Turbine ID")
        ax.set_xticklabels(ticklabels)

    if orient == "h": 
        ax = sns.barplot(x="ts", y="instanceID", hue = "value", data=df, orient = orient, order = order)
        plt.xlabel("Sample count")
        plt.ylabel("AWI Turbine ID")
        ax.set_yticklabels(ticklabels)
    
    ax.legend(title="Full performance flag")
    plt.show()


def clean_data(df):
    '''
    this function removes invalid values and remove rows with null entries  
    input: 
        df(dataframe): the dataframe to clean
    output: 
        df(dataframe): the clean dataframe
    '''    
    # power values must be >=0
    df = df[(df['Power'] >= 0)]

    # wind speed values must be >=0
    df = df[(df['Wind_speed'] >= 0)]
    df = df[~df['Wind_speed'].isnull()]

    if df['TI'].isnull().all(): 
        pass
    else:
        # TI values must be in [0, 100]
        df = df[(df['TI'] >= 0) & (df['TI'] <= 100)]
        df = df[~df['TI'].isnull()]

    if df['Temperature'].isnull().all(): 
        pass
    else:
        # temperature values must be in [-5, 40]
        df = df[(df['Temperature'] >= -5) & (df['Temperature'] <= 40)]
        df = df[~df['Temperature'].isnull()]
   
    return df


def plot_violinplot(df, var, var_name, order, ticklabels, figsize, orient = "v"):
    '''
    input: 
        df(dataframe): the dataframe to plot
        var(list): the list of variables to plot
        var_name(list): the list of the variable names to be displayed in the plot
        order(list): the list of the turbine IDs in order 
        xticklabels(list): the list of the turbine IDs name, to be displayed in the plot
        figsize(tuple): the figure size 
        orient(string): "v" for vertical orientation, "h" for horizontal orientation
    output: 
        none
    '''      
    fig, ax = plt.subplots(len(var), figsize = figsize)
    
    if orient == "v":
        for i in range (0, len(var)):
            sns.violinplot(data = df, y = var[i], x = 'instanceID', orient = orient, ax = ax[i], order = order)
            ax[i].set(ylabel = var_name[i], xlabel = 'turbine ID')
            ax[i].set_xticklabels(ticklabels)

    if orient == "h":
        for i in range (0, len(var)):
            sns.violinplot(data = df, x = var[i], y = 'instanceID', orient = orient, ax = ax[i], order = order)
            ax[i].set(xlabel = var_name[i], ylabel = 'turbine ID')
            ax[i].set_yticklabels(ticklabels)

    plt.show()


def plot_powercurve(df, order, figsize=(18,18)):
    '''
    input: 
        df(dataframe): the dataframe to plot
        order(list): the list of the turbine IDs in order
        figsize(tuple): the figure size 
    output: 
        none
    '''     
    fig, ax = plt.subplots(5,5, figsize=figsize, sharex='col', sharey='row');

    i = 0

    for r in range (5):
        for c in range (5):
            df_turbine = df[df['instanceID'] == order[i]]
            sns.scatterplot(x = df_turbine['Wind_speed'], y = df_turbine['Power'], ax = ax[r][c], s = 1, 
                            edgecolor = None)
            ax[r][c].set_title(order[i])
            ax[r][c].set_xlabel("Wind_speed") 
            ax[r][c].set_ylabel("Power") 
            i += 1
            if i >= len(order):
                break
        
        if i >= len(order):
            break
            
    plt.show()


def plot_vars(df, turbine_name, ws_range, x_var, figsize=(18,3)):
    '''
    input: 
        df(dataframe): the dataframe to plot
        turbine_name(string): the turbine ID to plot
        ws_range(tuple): the range of wind speed to plot, e.g. (1, 10) 
        x_var(list): the list of variables for the x-axis. if 3 variables are given, there will be 3 plots
        figsize(tuple): the figure size 
    output: 
        none
    '''   
    df_plot = df[(df['instanceID'] == turbine_name) & (df['Wind_speed'] >= ws_range[0]) & (df['Wind_speed'] <= ws_range[1])] 

    fig, ax = plt.subplots(1,len(x_var), figsize=(18,3), sharey='row')

    for c in range(len(x_var)):
        sns.scatterplot(x = df_plot[x_var[c]], y = df_plot['Power'], ax = ax[c], s = 1, edgecolor = None)
        ax[c].set_xlabel(x_var[c]) 
        ax[c].set_ylabel("Power") 

    plt.show()


def plot_TIeffect(df, turbine_name, ws_range1, ws_range2, figsize=(16,6)):
    '''
    input: 
        df(dataframe): the dataframe to plot
        turbine_name(string): the turbine ID to plot
        ws_range1(tuple): the range of wind speed for the plot on the left, e.g. (1, 2) 
        ws_range2(tuple): the range of wind speed for the plot on the right, e.g. (5, 6) 
        figsize(tuple): the figure size 
    output: 
        none
    '''     
    df_plot = df[(df['instanceID'] == turbine_name)]
    df_plot_left = df_plot[(df_plot['TI'] >= ws_range1[0]) & (df_plot['TI'] <= ws_range1[1])] 
    df_plot_right = df_plot[(df_plot['TI'] >= ws_range2[0]) & (df_plot['TI'] <= ws_range2[1])] 

    fig, ax = plt.subplots(1,2, figsize=(16,6), sharey='row')

    sns.scatterplot(x = df_plot['Wind_speed'], y = df_plot['Power'], ax = ax[0], s = 5, label = 'all data points', edgecolor = None)
    sns.scatterplot(x = df_plot_left['Wind_speed'], y = df_plot_left['Power'], ax = ax[0], s = 5 , label = 'data points where ' + str(ws_range1[0]) + '<= TI <= ' + str(ws_range1[1]), edgecolor = None)
    ax[0].set_xlabel('Wind speed, m/s') 
    ax[0].set_ylabel("Power, kW")  
    
    sns.scatterplot(x = df_plot['Wind_speed'], y = df_plot['Power'], ax = ax[1], s = 5, label = 'all data points', edgecolor = None)
    sns.scatterplot(x = df_plot_right['Wind_speed'], y = df_plot_right['Power'], ax = ax[1], s = 5, label = 'data points where ' + str(ws_range2[0]) + '<= TI <= ' + str(ws_range2[1]), edgecolor = None)
    ax[1].set_xlabel('Wind speed, m/s') 
    ax[1].set_ylabel("Power, kW") 

    plt.show()