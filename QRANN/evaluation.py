import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def eval_prediction(df):
    '''
    calculates the RMSE and MSE
    '''
    # RMSE - root mean square error
    # MAE - mean absolute error 
    RMSE = np.sqrt(metrics.mean_squared_error(df['Power'], df['Predicted_power']))
    MAE = metrics.mean_absolute_error(df['Power'], df['Predicted_power'])

    return RMSE, MAE

def eval_PI(df):
    '''
    calculates the PICP, NMPIW, and CWC
    '''
    # PICP - prediction interval coverage probability 
    # NMPIW - normalized mean prediction interval width 
    # CWC - coverage width base criterion
    num_outlier = len(df[(df['Under_pred'] == 1) | (df['Over_pred'] == 1)])
    PICP = (1 - num_outlier/len(df))
    max_power = df['Power'].max()
    min_power = df['Power'].min()
    NMPIW = df['PI_width'].mean()/(max_power-min_power)
    mu = 1-0.05
    gamma = PICP < mu
    CWC = NMPIW * (1 + (gamma * PICP * np.exp(-10*(PICP-mu))))

    return PICP, NMPIW, CWC

def PItrend(var, df):
    '''
    this function computes the NMPIW in bins. 
    input:
        var(string): the predictor variable to bin - "Wind_speed", "TI", or "Temperature"
        df(dataframe): the dataframe storing the results
    output: 
        df_trend(dataframe): the dataframe with NMPIW in bins 
    '''
    # define the bin intervals based on variable specified 
    if var == 'Wind_speed': 
        var_start = np.linspace(1, 29.5, 58)
        var_end = np.linspace(1.5, 30, 58)
        var_bin = ['1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5', '4.5-5', '5-5.5', '5.5-6',
                    '6-6.5', '6.5-7', '7-7.5', '7.5-8', '8-8.5', '8.5-9', '9-9.5', '9.5-10', '10-10.5', '10.5-11', 
                    '11-11.5', '11.5-12', '12-12.5', '12.5-13', '13-13.5', '13.5-14', '14-14.5', '14.5-15', '15-15.5', '15.5-16',
                    '16-16.5', '16.5-17', '17-17.5', '17.5-18', '18-18.5', '18.5-19', '19-19.5', '19.5-20', '20-20.5', '20.5-21',
                    '21-21.5', '21.5-22', '22-22.5', '22.5-23', '23-23.5', '23.5-24', '24-24.5', '24.5-25', '25-25.5', '25.5-26',
                    '26-26.5', '26.5-27', '27-27.5', '27.5-28', '28-28.5', '28.5-29', '29-29.5', '29.5-30']

    elif var == "TI": 
        var_start = np.linspace(0, 95, 20)
        var_end = np.linspace(5, 100, 20)
        var_bin = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', 
                    '50-55', '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100']
    
    elif var == "Temperature":
        var_start = np.linspace(-5, 37.5, 18)
        var_end = np.linspace(-2.5, 40, 18)
        var_bin = ['-5--2.5', '-2.5-0', '0-2.5', '2.5-5', '5-7.5', '7.5-10', '10-12.5', '12.5-15', '15-17.5', '17.5-20',
                            '20-22.5', '22.5-25', '25-27.5', '27.5-30', '30-32.5', '32.5-35', '35-37.5', '37.5-40']

    # calculate the range of power output (required to compute NMPIW)
    R = df['Power'].max() - df['Power'].min()

    # create the dataframe to store the binned NMPIW
    df_trend = pd.DataFrame()

    # loop over each bin
    for i in range (len(var_bin)):
        # filter the data
        temp_data = df[(df[var] >= var_start[i]) & (df[var] < var_end[i])]

        sample_count = len(temp_data)
        if (sample_count == 0): 
            RMSE = np.nan 
            PI_width_mean = np.nan 
            NMPIW = np.nan
        
        else:
            # compute the NMPIW
            PI_width_mean = temp_data['PI_width'].mean()
            NMPIW = PI_width_mean/R
        
        # combine them into  the dataframe df_trend
        df_temp = pd.DataFrame([var_bin[i], sample_count, NMPIW], 
                                index = ['Bin', 'Sample_count', 'NMPIW']).T
        if df_trend.empty:
            df_trend = df_temp

        else:
            df_trend = pd.concat([df_trend, df_temp])

    df_trend = df_trend.reset_index(drop=True)
    df_trend = df_trend.dropna(axis = 0)

    return df_trend 

def plot_PItrend(var, df, ylim = 0.2):
    '''
    this function plots the trend of NMPIW given the predictor variable. 
    input:
        var(string): the predictor variable to bin - "Wind_speed", "TI", or "Temperature"
        df(dataframe): the dataframe storing the results
        ylim(float): the upper limit of y axis, in the range of [0.0, 1.0]
    '''
    fig, ax = plt.subplots(figsize = (16,4))
    ax.plot(df['NMPIW'], marker = '*')
    ax.axes.set_xticks(ticks = df.index.values)
    ax.axes.set_xticklabels(df.iloc[:, 0].to_list())
    ax.set_xlabel(var)
    plt.xticks(rotation = -45)
    plt.ylim([0, ylim])
    plt.ylabel('Normalized mean PI width')
    plt.show()

def pivot(df, L0_threshold, L1_threshold, L2_threshold): 
    '''
    this function pivots the results and returns three dataframes corresponding to the three categories of outliers. 
    input: 
        df(dataframe): the results 
        L0_threshold, L1_threshold, L2_threshold (floats): the threshold values for each outlier category, in ratio [0.0, 1.0]
    output: 
        df_pivot_cat_1, df_pivot_cat_2, df_pivot_cat_3 (dataframes): the pivotted dataframes based on the category of outliers 
    '''

    # pivot
    df_pivot = pd.pivot_table(df, values=['Sample_count', 'Outlier_pred', 'Under_pred', 'Over_pred'], 
                                   index=['instanceID', 'Week'],
                                   aggfunc={'Sample_count': np.sum, 
                                            'Outlier_pred': np.sum, # category 1
                                            'Over_pred': np.sum, # category 2
                                            'Under_pred': np.sum}) # category 3

    # drop rows with insignificant sample count 
    min_samples =  252
    idx = df_pivot[df_pivot['Sample_count'] < min_samples].index
    df_pivot = df_pivot.drop(idx)

    # assign alert level based on threshold value
    df_pivot['Outlier_ratio'] = df_pivot['Outlier_pred']/df_pivot['Sample_count']
    df_pivot['Over_ratio'] = df_pivot['Over_pred']/df_pivot['Sample_count']
    df_pivot['Under_ratio'] = df_pivot['Under_pred']/df_pivot['Sample_count']

    df_pivot['Level_0'] = df_pivot['Outlier_ratio'] <= L0_threshold
    df_pivot['Level_1'] = (df_pivot['Outlier_ratio'] > L0_threshold) & (df_pivot['Outlier_ratio'] <= L1_threshold)
    df_pivot['Level_2'] = (df_pivot['Outlier_ratio'] > L1_threshold) & (df_pivot['Outlier_ratio'] <= L2_threshold)
    df_pivot['Level_3'] = df_pivot['Outlier_ratio'] > L2_threshold

    df_pivot['Level_0_up'] = df_pivot['Over_ratio'] <= L0_threshold/2
    df_pivot['Level_1_up'] = (df_pivot['Over_ratio'] > L0_threshold/2) & (df_pivot['Over_ratio'] <= L1_threshold/2)
    df_pivot['Level_2_up'] = (df_pivot['Over_ratio'] > L1_threshold/2) & (df_pivot['Over_ratio'] <= L2_threshold/2)
    df_pivot['Level_3_up'] = df_pivot['Over_ratio'] > L2_threshold/2

    df_pivot['Level_0_lo'] = df_pivot['Under_ratio'] <= L0_threshold/2
    df_pivot['Level_1_lo'] = (df_pivot['Under_ratio'] > L0_threshold/2) & (df_pivot['Under_ratio'] <= L1_threshold/2)
    df_pivot['Level_2_lo'] = (df_pivot['Under_ratio'] > L1_threshold/2) & (df_pivot['Under_ratio'] <= L2_threshold/2)
    df_pivot['Level_3_lo'] = df_pivot['Under_ratio'] > L2_threshold/2

    df_pivot['Level_combined'] = df_pivot['Level_1']*1 + df_pivot['Level_2']*2 + df_pivot['Level_3']*3
    df_pivot['Level_combined_up'] = df_pivot['Level_1_up']*1 + df_pivot['Level_2_up']*2 + df_pivot['Level_3_up']*3
    df_pivot['Level_combined_lo'] = df_pivot['Level_1_lo']*1 + df_pivot['Level_2_lo']*2 + df_pivot['Level_3_lo']*3

    df_pivot = df_pivot.reindex(columns = ['Level_combined', 'Level_combined_up', 'Level_combined_lo'])
    df_pivot = df_pivot.reset_index()

    # create dummy rows to ensure all weeks appear in heatmap 
    for week in range (1, 54): 
        if df_pivot[(df_pivot['Week'] == week)].empty: 
            new_row = {'instanceID': site + '_WTG01', 'Week':week, 'Level_combined':np.nan, 'Level_combined_up':np.nan, 
                    'Level_combined_lo':np.nan}
            df_pivot = df_pivot.append(new_row, ignore_index = True)

    # create separate dataframe for outliers at either/ above UQ/ below LQ
    df_pivot_cat_1 = df_pivot.pivot(index = 'instanceID', columns = 'Week', values = 'Level_combined')
    df_pivot_cat_2 = df_pivot.pivot(index = 'instanceID', columns = 'Week', values = 'Level_combined_up')
    df_pivot_cat_3 = df_pivot.pivot(index = 'instanceID', columns = 'Week', values = 'Level_combined_lo')

    return df_pivot_cat_1, df_pivot_cat_2, df_pivot_cat_3


def plot_heatmap(df_cat, category): 
    '''
    plots a heatmap showing the weekly alert levels for each turbine. 
    input: 
        df_cat(dataframe): the pivotted dataframe
        category(string): the category of the outlier, which will be displayed as the title of the heatmap
    '''
    # define colour map
    colors = [(0, 0.85, 0), (1, 1, 0), (1, 0.7, 0), (1, 0, 0)]
    n_bins = 4
    cmap_name = 'cm'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    fig, ax = plt.subplots(figsize = (20,8))
    ax = sns.heatmap(df_cat, vmin = 0, vmax = 4, annot = True, cmap = cm, cbar = True, 
                linewidths = 1)

    # customize colour bar settings
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    colorbar.set_ticklabels([0, 1, 2, 3])
    
    plt.yticks(rotation=0) 
    plt.title(category)
    plt.show()


def plot_time(site, turbine, week, df_target, df_outlier, ylim = 2250):
    '''
    plots the measured power, upper quantile, and lower quantile of the specified turbine and week number
    input: 
        site(string): the name of the site 
        turbine(string): the turbine number 
        week(int): the week number
        df_target(dataframe): the dataframe filtered based on the turbine number and week number
        df_outlier(dataframe): the outliers identified in df_target
        ylim(int): the upper limit of the y axis, power 
    '''
    fig, ax = plt.subplots(figsize = (16, 4))
    ax.plot(df_target['ts'], df_target['Power'], label = 'Measured power')
    ax.plot(df_target['ts'], df_target['UQ'], label = 'Upper Quantile of PI')
    ax.plot(df_target['ts'], df_target['LQ'], label = 'Lower Quantile of PI')
    ax.scatter(df_outlier['ts'], df_outlier['Power'], color = 'red', label = 'outlier')
    plt.xlabel('Time')
    plt.ylabel('Power, kW')
    plt.title(site + '_WTG' + turbine + ' Week ' + str(week) + ' (using PIs)')
    plt.ylim([-100, ylim])
    plt.legend()
    plt.show()


def plot_2pcurve(site, turbine, week, df_turbine, df_target, df_outlier):
    '''
    plots the power curves showing the outliers identified for a particular turbine in a specified week
    input: 
        site(string): the name of the site 
        turbine(string): the turbine number 
        week(int): the week number
        df_turbine(dataframe): the dataframe filtered based on the turbine number
        df_target(dataframe): the dataframe filtered based on the turbine number and week number
        df_outlier(dataframe): the outliers identified in df_target
    '''
    colors = np.array([(0.70, 0.85, 1), (0, 0.40, 0.80), (0.80, 0, 0)])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))
    
    ax1.scatter(df_turbine['Wind_speed'], df_turbine['Power'], s = 2, label = site + '_WTG' + turbine, 
                color = colors[0])
    ax1.scatter(df_target['Wind_speed'], df_target['Power'], s = 5, label = 'Week ' + str(week), 
                color = colors[1])
    ax1.set_xlabel('Wind speed, m/s')
    ax1.set_ylabel('Power, kW')
    ax1.legend(loc = 'lower right')
    ax1.grid()
    
    ax2.scatter(df_turbine['Wind_speed'], df_turbine['Power'], s = 2, label = site + "_WTG" + turbine, 
                color = colors[0])
    ax2.scatter(df_outlier['Wind_speed'], df_outlier['Power'], s = 5, label = 'Week ' + str(week) + ' Outlier', 
                color = colors[2])
    ax2.set_xlabel('Wind speed, m/s')
    ax2.set_ylabel('Power, kW')
    ax2.legend(loc = 'lower right')
    ax2.grid()
    plt.show()


def plot_variables (week, df_turbine, df_target, df_outlier):
    '''
    plots the individual variables of the filtered dataframe 
    input: 
        week(int): the week number
        df_turbine(dataframe): the dataframe filtered based on the turbine number
        df_target(dataframe): the dataframe filtered based on the turbine number and week number
        df_outlier(dataframe): the outliers identified in df_target
    '''
    plt.subplots(figsize = (16, 2))
    plt.plot(df_target['ts'], df_target['Temperature'])
    plt.scatter(df_outlier['ts'], df_outlier['Temperature'], label = 'Week ' + str(week) + ' Outlier', 
                color = 'red')
    plt.title('Temperature')
    plt.ylim((df_turbine['Temperature'].min(), df_turbine['Temperature'].max()))
    plt.ylabel('Temperature')
    plt.legend()
    
    plt.subplots(figsize = (16, 2))
    plt.plot(df_target['ts'], df_target['TI'])
    plt.scatter(df_outlier['ts'], df_outlier['TI'], label = 'Week ' + str(week) + ' Outlier', 
                color = 'red')
    plt.title('TI')
    plt.ylim((df_turbine['TI'].min(), df_turbine['TI'].max()))
    plt.ylabel('TI')
    plt.legend()
    
    plt.subplots(figsize = (16, 2))
    plt.plot(df_target['ts'], df_target['Wind_speed'])
    plt.scatter(df_outlier['ts'], df_outlier['Wind_speed'], label = 'Week ' + str(week) + ' Outlier', 
                color = 'red')
    plt.title('Wind speed')
    plt.ylim((df_turbine['Wind_speed'].min(), df_turbine['Wind_speed'].max()))
    plt.ylabel('Wind speed')
    plt.legend()
    
    plt.show()


def plot_res(df, df_outlier, ylim = 400, outlier = True):
    plt.rc('font', size=12)
    plt.subplots(figsize = (7,6))
    sns.scatterplot(data = df, x = df.index.values, y = df['Error'], s=2, label = 'power residuals')
    if outlier == True:
            sns.scatterplot(data = df_outlier, x = df_outlier.index.values, y = df_outlier['Error'], color = 'red', s=2, label = 'power residuals falling outside the PI')
    plt.ylim(-max(abs(df.Error.max()), abs(df.Error.min())), 
            max(abs(df.Error.max()), abs(df.Error.min())))
    plt.xlabel('Sample number')
    plt.ylabel('Power residual, kW')
    plt.ylim([-ylim, ylim])
    plt.legend()
    plt.show()


def plot_reshistogram(df, xlim = 400, ylim = 10000):
    '''
    plots the residual histogram
    '''
    plt.rc('font', size=12)
    plt.subplots(figsize = (7, 6))
    sns.histplot(data = df, x = 'Error', kde = True, bins = 50)
    plt.xlim(-max(abs(df.Error.max()), abs(df.Error.min())), 
            max(abs(df.Error.max()), abs(df.Error.min())))
    plt.ylim([0, ylim])
    plt.xlim([-(xlim), xlim])
    plt.xlabel('Power residual, kW')
    plt.ylabel('Sample count')
    plt.show()


def plot_pcurve(df, df_outlier, outlier = True):
    plt.rc('font', size=12)
    plt.subplots(figsize = (7, 6))
    sns.scatterplot( data = df, x = 'Wind_speed', y = 'Power', s = 2, label = 'all data points')
    if outlier == True:
        sns.scatterplot( data = df_outlier, x = 'Wind_speed', y = 'Power', s = 2, label = 'outliers identified by PI', color = "Red")
    plt.xlabel('Wind speed, m/s')
    plt.ylabel('Power, kW')
    plt.legend()
    plt.show()


def plot_res_vs_var(df, df_outlier, var, ylim = 400, outlier = True):   
    plt.rc('font', size=12)
    plt.subplots(figsize = (7, 6))
    sns.scatterplot(data = df, x = var, y = "Error", s = 2)
    if outlier == True:
        sns.scatterplot(data = df_outlier, x = var, y = "Error", s = 2, color = "red", label = 'outliers identified by PI')
    plt.ylim([-ylim, ylim])
    plt.xlabel(var)
    plt.ylabel('Power residual, kW')
    plt.legend()
    plt.show()    