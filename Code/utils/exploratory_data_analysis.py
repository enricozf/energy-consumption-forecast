#%%
import pandas as pd 
import numpy as np
from os import path, walk
from tensorflow.signal import rfft

import matplotlib.pyplot as plt
import seaborn as sns

GRAPHIC_COLORS = {
    'Affluent' : 'cornflowerblue',
    'Comfortable' : 'mediumseagreen',
    'Adversity' : 'goldenrod',
    'ACORN-U' : 'slategray',
    'weather' : 'indianred'
}

def read_acorn_group_blocks(
    blocks_path : str ='..//Data//daily_dataset'):

    # Create empty DataFrame
    df_acorn = pd.DataFrame()

    # Concatenate each block from particular acorn group
    for _, _, files in walk(blocks_path):
        for file in files:
            print('Reading {}...'.format(file))
            block_filename = path.normpath(path.join(blocks_path, file))

            df_block = pd.read_csv(block_filename, header=0)
            df_acorn = pd.concat([df_acorn, df_block])
    
    print('Done.')
    df_acorn['day'] = pd.to_datetime(df_acorn['day'])
    return df_acorn.set_index(['LCLid', 'day'])

def split_into_acorns(
    df_all : pd.DataFrame,
    df_info : pd.DataFrame,
    ):
    # Remove all SM that are taxed as ToU
    not_tou_sms_idx = df_info[df_info['stdorToU'] == 'Std']['LCLid'].values
    df_wot_tou = df_all.loc[not_tou_sms_idx]
    
    # Seggregate data into acorn groups
    dict_acorns_data = {}
    for acorn in df_info['Acorn_grouped'].unique():
        acorn_data_idx = df_info[df_info['Acorn_grouped']==acorn]['LCLid'].values
        dict_acorns_data[acorn] = df_wot_tou.loc[acorn_data_idx]

    return dict_acorns_data

def bar_plot(
    ax : plt.Axes,
    values : list,
    classes : list,
    colors : list,
    xlabel : str,
    ylabel : str):

    ax.barh(y=range(len(classes)), width=values, 
           tick_label=classes,
           color=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0,0), 1, 1, color=c) for c in colors]
    lgd = ax.legend(handles=handles, labels=classes, 
                    bbox_to_anchor=[1.01, 1], loc=2, title='Legenda')
    return lgd

def plot_std_or_tou_difference(
    df_info : pd.DataFrame,
    stdortou_col = 'stdorToU',
    save_path : str = '..//Figures//',
    save_filename : str = None,
    **kwargs):
    # Create figure and axis
    fig, ax = plt.subplots(**kwargs)

    # Create dict of these types
    dict_diff = {k : v for k, v in df_info[stdortou_col].value_counts().items()}

    colors = ['cornflowerblue', 'indianred']

    # ax.barh(y=range(len(dict_diff.keys())), width=dict_diff.values(), 
    #        tick_label=dict_diff.keys(),
    #        color=colors)
    # ax.set_xlabel('Consumo Energético [kWh]')
    # ax.set_ylabel('Acorns da pesquisa')
    # handles = [plt.Rectangle((0,0), 1, 1, color=c) for c in colors]
    # ax.legend(handles=handles, labels=dict_diff.keys(), 
    #           bbox_to_anchor=[1.01, 1], loc=2, title='Legenda')
    lgd = bar_plot(ax, values=list(dict_diff.values()), 
                   classes=list(dict_diff.keys()), 
                   colors=colors, 
                   ylabel='Tipo de tarifa do medidor',
                   xlabel='Número de medidores')

    if save_filename:
        save_filename = path.normpath(path.join(save_path, save_filename))
        plt.savefig(save_filename, bbox_extra_artist=(lgd,), bbox_inches='tight')
    else:
        return fig, ax

def plot_total_energy_per_acorn(
    dict_data : dict,
    lst_acorn_names : list,
    sum_col : str = 'energy_sum',
    save_path : str = '..//Figures//',
    save_filename : str = None,
    **kwargs):
    # Create figure and axis
    fig, ax = plt.subplots(**kwargs)

    # Create list of total energy
    total_per_acorn = [dict_data[acorn][sum_col].sum() for acorn \
                                                        in lst_acorn_names]

    # colors = ['cornflowerblue', 'mediumseagreen', 'goldenrod', 'indianred']
    colors = [GRAPHIC_COLORS.get(acorn) for acorn in lst_acorn_names]
    # ax.barh(y=range(len(lst_acorn_names)), width=total_per_acorn, 
    #        tick_label=lst_acorn_names,
    #        color=colors)
    # ax.set_xlabel('Consumo Energético [kWh]')
    # ax.set_ylabel('Acorns da pesquisa')
    # handles = [plt.Rectangle((0,0), 1, 1, color=c) for c in colors]
    # ax.legend(handles=handles, labels=lst_acorn_names, 
    #           bbox_to_anchor=[1.01, 1], loc=2, title='Legenda')
    lgd = bar_plot(ax, values=total_per_acorn, classes=lst_acorn_names, 
                   colors=colors, 
                   ylabel='Acorns da pesquisa',
                   xlabel='Consumo Energético [kWh]')

    if save_filename:
        save_filename = path.normpath(path.join(save_path, save_filename))
        plt.savefig(save_filename, bbox_extra_artist=(lgd,), bbox_inches='tight')
    else:
        return fig, ax

def plot_total_relative_energy_per_acorn(
    dict_data : dict,
    lst_acorn_names : list,
    sum_col : str = 'energy_sum',
    count_col :str = 'energy_count',
    save_path : str = '..//Figures//',
    save_filename : str = None,
    **kwargs):
    # Create figure and axis
    fig, ax = plt.subplots(**kwargs)

    # Create list of total energy
    relative_total_per_acorn = [dict_data[acorn][sum_col].sum() / \
                                dict_data[acorn][count_col].sum() for acorn \
                                                            in lst_acorn_names]

    # colors = ['cornflowerblue', 'mediumseagreen', 'goldenrod', 'indianred']
    colors = [GRAPHIC_COLORS.get(acorn) for acorn in lst_acorn_names]
    # ax.barh(y=range(len(lst_acorn_names)), width=total_per_acorn, 
    #        tick_label=lst_acorn_names,
    #        color=colors)
    # ax.set_xlabel('Consumo Energético Relativo [kWh/#medições]')
    # ax.set_ylabel('Acorns da pesquisa')
    # handles = [plt.Rectangle((0,0), 1, 1, color=c) for c in colors]
    # ax.legend(handles=handles, labels=lst_acorn_names, 
    #           bbox_to_anchor=[1.01, 1], loc=2, title='Legenda')
    lgd = bar_plot(ax, values=relative_total_per_acorn, classes=lst_acorn_names, 
                   colors=colors, 
                   ylabel='Acorns da pesquisa',
                   xlabel='Consumo Energético Relativo [kWh/(número medições)]')

    if save_filename:
        save_filename = path.normpath(path.join(save_path, save_filename))
        plt.savefig(save_filename, bbox_extra_artist=(lgd,), bbox_inches='tight')
    else:
        return fig, ax

def correlate_energy_weather_data(
    dict_data : dict,
    lst_acorn_names : list,
    df_weather : pd.DataFrame,
    weather_col : str,
    sum_col : str = 'energy_sum',
    count_col : str = 'energy_count',
    day_col : str = 'day',
    save_path : str = '..//Figures//',
    save_filename : str = None,
    invert_weather_curve : bool = False,
    **kwargs):
    # Create figure and axis
    fig, ax = plt.subplots(**kwargs)
    
    # Get relative consumption values per acorn
    rel_energy_per_acorn = {
        acorn : dict_data[acorn][sum_col].groupby(day_col).sum() / \
                dict_data[acorn][count_col].groupby(day_col).sum() for acorn in \
                                                                lst_acorn_names
    }
    for k, v in rel_energy_per_acorn.items():
        rel_energy_per_acorn[k].index = pd.to_datetime(v.index)
    
    # Get weather data
    sr_weather = df_weather[weather_col]
    # Invert waether data curve
    sr_weather = sr_weather*(-1)**int(invert_weather_curve)

    # Get colors from list
    # colors = ['cornflowerblue', 'mediumseagreen', 'goldenrod', 'indianred']
    colors = [GRAPHIC_COLORS.get(acorn) for acorn in lst_acorn_names]

    # Plot consumption and weather data
    for acorn, color in zip(lst_acorn_names, colors):
        ax.plot(rel_energy_per_acorn[acorn].sort_index(), color=color)
    ax2 = ax.twinx()
    ax2.plot(sr_weather.sort_index(), color=GRAPHIC_COLORS.get('weather'),
             linewidth=.6)

    ax.set_xlabel('Data')
    ax.set_ylabel('Consumo Energético Relativo [kWh/(número medições)]')
    ax2.set_ylabel(weather_col)
    # TODO: set x limits from this image, using energy data min and max dates
    x_lims = (max([rel_energy_per_acorn[ac].index.min()] for ac in\
                                                         lst_acorn_names),
              min([rel_energy_per_acorn[ac].index.max()] for ac in\
                                                         lst_acorn_names))
    ax.set_xlim(x_lims)
    handles = [plt.Rectangle((0,0), 1, 1, color=c) for c in \
                                colors+[GRAPHIC_COLORS.get('weather')]]
    lgd = ax.legend(handles=handles, labels=lst_acorn_names+[weather_col], 
              bbox_to_anchor=[1.05, 1], loc=2, title='Acorns da pesquisa')

    if save_filename:
        save_filename = path.normpath(path.join(save_path, save_filename))
        plt.savefig(save_filename, bbox_extra_artist=(lgd,), bbox_inches='tight')
    else:
        return fig, ax

def plot_fourier_trsnfd_weather_data(sr_weather : pd.Series):
    
    fft = rfft(sr_weather)
    f_per_dataset = np.arange(0, len(fft))

    # Get number of half hour samples
    n_samples_hours = len(sr_weather)
    hours_per_year = 24*365.2524
    years_per_dataset = n_samples_hours / hours_per_year

    f_per_year = f_per_dataset / years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.ylim((-1,50000))
    plt.xscale('log')
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')

    return fft
    

#%%
if __name__ == '__main__':

    # Load smart meters information
    sm_info_file = '..//Data//informations_households.csv'
    df_sm_info = pd.read_csv(sm_info_file, header=0)

    # Load weather data
    weather_daily_file = '..//Data//weather_daily_darksky.csv'
    df_weather = pd.read_csv(weather_daily_file, header=0)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather = df_weather.set_index('time')

    # Load all data
    df_complete = read_acorn_group_blocks()

    # Remove ToU SMs and split it into accorns
    dict_data = split_into_acorns(df_complete, df_sm_info)

    # Generate images
    plot_std_or_tou_difference(df_sm_info, 
                               save_filename='Diff_Tarifas.png')

    plot_total_energy_per_acorn(
        dict_data, 
        lst_acorn_names=['Affluent', 'Comfortable', 'Adversity', 'ACORN-U'],
        save_filename='Energia_Total_por_Acorn.png')

    plot_total_relative_energy_per_acorn(
        dict_data, 
        lst_acorn_names=['Affluent', 'Comfortable','Adversity', 'ACORN-U'],
        save_filename='Energia_Relativa_Total_por_Acorn.png')
    
    for weather_col in df_weather.columns:
        print(weather_col)
        if 'time' in weather_col.lower() or 'summary' in weather_col.lower():
            continue
        correlate_energy_weather_data(
            dict_data,
            lst_acorn_names=['Affluent', 'Comfortable', 'Adversity'],
            df_weather=df_weather,
            weather_col=weather_col,
            save_filename='Relação_Consumo_X_{}.png'.format(weather_col),
            figsize=(10,6)
        )