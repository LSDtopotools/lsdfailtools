"""
Author: Marina Ruiz Sanchez-Oro
Date: 17/01/2022
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_timeseries_failure(input_csv):
    """
    plot_timeseries_failure plots the timeseries of factor of safety for a given location
    :param input_csv: csv file with the timeseries of factor of safety
    """
    timeseries_df = pd.read_csv(input_csv, index_col = None)
    timeseries_df['is_it_failure'] = np.where((timeseries_df['FoS']>=1),0,1)
    first_day_of_failure = (timeseries_df['FoS'] < 1.0).idxmax()
    last_failure = timeseries_df['is_it_failure'][timeseries_df['is_it_failure']==False].index[-1]



    fig, ax = plt.subplots()
    textstr = f'First possible failure on day {first_day_of_failure}'
    textstr = '\n'.join((
        f'First possible failure on day {first_day_of_failure}',
        f'Last possible failure on day {last_failure}'))

    props = dict(boxstyle='round', facecolor='white')

    ax.plot(timeseries_df['days'], timeseries_df['FoS'], label = 'FoS timeseries')
    plt.axhline(y=1., color='black', linestyle='--', label = 'FoS failure threshold (FoS = 1)')
    ax.axvspan(first_day_of_failure, last_failure, alpha=0.5, color='#FF0000', label = 'Possible failure interval')
    ax.text(0.98, 0.77, textstr,  fontsize=10,verticalalignment = 'top', horizontalalignment = 'right', bbox=props,transform=ax.transAxes)

    plt.xlabel('Days')
    plt.ylabel('Factor of Safety (FoS)')
    plt.legend(facecolor='white', edgecolor='black')
    base = os.path.basename(input_csv)
    file_name = os.path.splitext(base)[0]
    print(file_name)


    out_name = 'plot_'+file_name+'.png'
    print(out_name)
    plt.savefig(out_name)

######
# Example use with test file:
# plot_timeseries_failure('fos_timeseries_41.0776_15.0254.csv')
######
