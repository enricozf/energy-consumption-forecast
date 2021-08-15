#%%
from numpy.core.numeric import roll
import pandas as pd 
import numpy as np
from os import path, walk
from tensorflow.signal import rfft
from scipy.fftpack import fft, fftshift, ifft, ifftshift, fftfreq

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
    ylabel : str,
    **kwargs):

    ax.barh(y=range(len(classes)), width=values, 
           tick_label=classes,
           color=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0,0), 1, 1, color=c) for c in colors]
    lgd = ax.legend(handles=handles, labels=classes, 
                    bbox_to_anchor=[1.01, 1], loc=2, title='Legenda',
                    **kwargs)
    return lgd

def plot_std_or_tou_difference(
    df_info : pd.DataFrame,
    stdortou_col = 'stdorToU',
    save_path : str = '..//Figures//',
    save_filename : str = None,
    fontsize=10,
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
                   xlabel='Número de medidores',
                   title_fontsize=fontsize, prop={'size' : fontsize})

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
    fontsize=10,
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
                   xlabel='Consumo Energético [kWh]',
                   title_fontsize=fontsize, prop={'size' : fontsize})

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
    fontsize=10,
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
                   xlabel='Consumo Energético Relativo [kWh/(número medições)]',
                   title_fontsize=fontsize, prop={'size' : fontsize})

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
    fontsize=10,
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
              bbox_to_anchor=[1.05, 1], loc=2, title='Acorns da pesquisa',
              title_fontsize=fontsize, prop={'size' : fontsize})

    if save_filename:
        save_filename = path.normpath(path.join(save_path, save_filename))
        plt.savefig(save_filename, bbox_extra_artist=(lgd,), bbox_inches='tight')
    else:
        return fig, ax
    
def scatter_temperature_consumption(
    temperature_file_path : str ='..//Data//weather_daily_darksky.csv',
    consumption_file_path : str ='..//Data//daily_dataset',
    sm_info_file : str = '..//Data//informations_households.csv',
    lst_acorn_names : str = ['Affluent', 'Comfortable', 'Adversity'],
    sum_col : str = 'energy_sum',
    count_col : str = 'energy_count',
    day_col : str = 'day',
    weather_col : str ='temperatureMax',
    save_filename : str =None,
    save_path : str = '..//Figures//',
    fontsize=10):
    
    # Read all Smart meters data
    dict_data = read_acorn_group_blocks(consumption_file_path)
    # Load smart meters information
    df_sm_info = pd.read_csv(sm_info_file, header=0)
    # Split into acorns
    dict_data = split_into_acorns(dict_data, df_sm_info)

    acorn_data = {
        acorn : dict_data[acorn][sum_col].groupby(day_col).sum() / \
                dict_data[acorn][count_col].groupby(day_col).sum() \
                    for acorn in lst_acorn_names
    }
    # Read weather temperature data
    weather_data = pd.read_csv(temperature_file_path, header=0, index_col='time')
    weather_data = weather_data[weather_col].sort_index()
    weather_data.index = pd.to_datetime(weather_data.index)

    earlier_data_day = max([acorn_data[a].index.min()] for a in lst_acorn_names)
    earlier_data_day = earlier_data_day[0]
    later_data_day = min([acorn_data[a].index.max()] for a in lst_acorn_names)
    later_data_day = later_data_day[0]

    # Truncate data
    weather_data = weather_data.loc[str(earlier_data_day):str(later_data_day)]

    # Plotting data
    fig, ax = plt.subplots(figsize=(10,6))
    colors = []
    for a in lst_acorn_names:
        acorn_data[a] = acorn_data[a].loc[earlier_data_day:later_data_day]
        ax.scatter(1*weather_data, acorn_data[a].sort_index(), 
                   cmap=GRAPHIC_COLORS[a])
        colors.append(GRAPHIC_COLORS.get(a))
    
    handles = [plt.Rectangle((0,0), 1, 1, color=c) for c in colors]
    lgd = ax.legend(handles=handles, labels=lst_acorn_names, 
              bbox_to_anchor=[1.05, 1], loc=2, title='Acorns da pesquisa',
              title_fontsize=fontsize, prop={'size' : fontsize})
    ax.set_ylabel('Consumo Energético Relativo [kWh/(número medições)]')
    ax.set_xlabel('Temperatura (°C)')

    if save_filename:
        save_filename = path.normpath(path.join(save_path, save_filename))
        plt.savefig(save_filename, bbox_extra_artist=(lgd,), bbox_inches='tight')
    else:
        return fig, ax

def show_week_day_consumption_diff(
    consumption_file_path : str = '..//Data//daily_dataset',
    sm_info_file : str = '..//Data//informations_households.csv',
    lst_acorn_names : str = ['Affluent', 'Comfortable', 'Adversity'],
    sum_col : str = 'energy_sum',
    count_col : str = 'energy_count',
    save_filename : str = None,
    save_path : str = '..//Figures//',
    fontsize=10):
    
    # Read all Smart meters data
    df_complete = read_acorn_group_blocks(consumption_file_path)
    df_complete['weekday'] = df_complete.reset_index()['day'].\
                                apply(lambda x : x.weekday()).values
    # Load smart meters information
    df_sm_info = pd.read_csv(sm_info_file, header=0)
    # Split into acorns
    dict_data = split_into_acorns(df_complete, df_sm_info)

    acorn_data = {
        acorn : dict_data[acorn].groupby('weekday')[sum_col].sum() / \
                dict_data[acorn].groupby('weekday')[count_col].sum() \
                    for acorn in lst_acorn_names
    }
    
    fig, ax = plt.subplots(figsize=(10,6))
    bottom = np.zeros(len(acorn_data[lst_acorn_names[0]]))
    idx = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
    colors = []
    for acorn in lst_acorn_names:
        acorn_data[acorn] = acorn_data[acorn].sort_index()
        ax.bar(idx, acorn_data[acorn], bottom=bottom,
                color=GRAPHIC_COLORS[acorn], label=acorn)
        bottom += acorn_data[acorn]
        colors.append(GRAPHIC_COLORS.get(acorn))
    
    handles = [plt.Rectangle((0,0), 1, 1, color=c) for c in colors]
    lgd = ax.legend(handles=handles, labels=lst_acorn_names, 
              bbox_to_anchor=[1.05, 1], loc=2, title='Acorns da pesquisa',
              title_fontsize=fontsize, prop={'size' : fontsize})
    ax.set_ylabel('Consumo Energético Relativo [kWh/(número medições)]',
                  fontsize=fontsize)
    ax.set_xlabel('Dias da semana', fontsize=fontsize)

    if save_filename:
        save_filename = path.normpath(path.join(save_path, save_filename))
        plt.savefig(save_filename, bbox_extra_artist=(lgd,), bbox_inches='tight')
    else:
        return fig, ax

def find_closest_2base_power(target):
    expoent = 0
    while target > 2**expoent:
        expoent+=1

    return 2**expoent

def plot_fourier_trsnfd_weather_data(sr_weather : pd.Series, r_size : int):
    
    # fft = rfft(sr_weather)
    fft_signal = fftshift(fft(sr_weather))
                            #   find_closest_2base_power(sr_weather.size)))
    f_per_dataset = np.arange(0, len(fft_signal))

    # Get number of half hour samples
    n_samples_hours = len(sr_weather)
    counts_per_year = 24*365.2524 / r_size
    years_per_dataset = n_samples_hours / counts_per_year

    f_per_year = f_per_dataset / years_per_dataset
    plt.step(f_per_year, np.abs(fft_signal)) 
    # plt.ylim((-1,50000))
    plt.xscale('log')
    # plt.xlim([0.1, max(plt.xlim())])
    # plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')

    return fft_signal
    
def restore_fft_signal(
    fft_signal : np.ndarray, 
    time_arr : np.ndarray,
    shift_flag : bool = True,):
    signal_copy = fft_signal.copy()
    signal_len = signal_copy.size
    if shift_flag: signal_copy = ifftshift(signal_copy)

    components = [signal_copy[k]*np.exp(2j*np.pi*k*time_arr/signal_len) \
                    for k in range(signal_len) if signal_copy[k]!= 0]
    return np.vstack(components).sum(axis=0)/signal_len

def represent_temperature_fft(
    weather_daily_file = '..//Data//weather_hourly_darksky.csv',
    desired_col = 'temperature',
    time_col = 'time',
    save_original_filename=None,
    save_representation_filename=None,
    save_path='..//Figures//',
    fontsize='10'):
    # Load weather daily data
    df_weather = pd.read_csv(weather_daily_file, header=0)
    df_weather[time_col] = pd.to_datetime(df_weather[time_col])
    df_weather = df_weather.set_index(time_col)
    thrs = 80
    
    # Get weekly mean values
    # display(df_weather[desired_col].head())
    rolling_size = 24*7
    weather_data = df_weather[desired_col].sort_index().rolling(rolling_size).\
                    mean()[::rolling_size].dropna()
    print('Rolling size: ', rolling_size, ' days.')

    # Plot weather signal in frequency domain
    plt.figure(figsize=(10,6))
    fft_signal = plot_fourier_trsnfd_weather_data(weather_data.values,
                                                  rolling_size)
    xlim = np.arange(plt.gca().get_xlim()[0],plt.gca().get_xlim()[1])
    plt.plot(xlim, [thrs]*xlim.size) # plot threshold
    # plt.gca().set_ylim((0, 500))

    # Plot original data
    _, ax = plt.subplots(figsize=(10,6))
    ax.plot(weather_data, 
            label='Média semanal amostrada') # Weekly mean data with step / strides
    ax.plot(df_weather[desired_col].sort_index().rolling(rolling_size).mean(),
             color='indianred', alpha=.7,
             label='Média semanal') # Weekly mean data
    ax.plot(df_weather[desired_col].sort_index(), color='cornflowerblue', 
             alpha=.2, label='Dados originais') # Original data
    ax.set_ylabel('Temperatura (°C)', fontsize=fontsize)
    ax.set_xlabel('Data', fontsize=fontsize)
    lgd = ax.legend( 
              bbox_to_anchor=[1.05, 1], loc=2, title_fontsize=fontsize,
              title='Descrição curvas', prop={'size':fontsize}
              )
    
    if save_original_filename:
        save_filename = path.normpath(path.join(save_path, save_original_filename))
        plt.savefig(save_filename, bbox_extra_artist=(lgd,), bbox_inches='tight')

    # Get signal with filtered frequencies
    fft_signal2 = fft_signal.copy()
    fft_signal2[np.abs(fft_signal) <= thrs] = 0
    _, ax = plt.subplots(figsize=(10,6))
    # plt.plot(ifft(ifftshift(fft_signal2)),color='cornflowerblue')
    # Highly filtered signal
    ax.plot(
        ifft(ifftshift(np.where(np.abs(fft_signal)>150, fft_signal, 0))),
        color='mediumseagreen', label='150'
    )
    # Best filtered signal
    ax.plot(
        ifft(ifftshift(np.where(np.abs(fft_signal)>thrs, fft_signal, 0))),
        color='cornflowerblue', label=str(thrs)
    )
    # Poorly filtered signal
    ax.plot(
        ifft(ifftshift(np.where(np.abs(fft_signal)>50, fft_signal, 0))),
        color='indianred', label='50'
    )
    ax.set_ylabel('Temperatura (°C)', fontsize=fontsize)
    ax.set_xlabel('Número de semanas', fontsize=fontsize)
    lgd = ax.legend( 
              bbox_to_anchor=[1.05, 1], loc=2, title_fontsize=fontsize,
              title='Limiares de frequência', prop={'size':fontsize}
              )
    
    if save_representation_filename:
        save_filename = path.normpath(path.join(
            save_path, save_representation_filename))
        plt.savefig(save_filename, bbox_extra_artist=(lgd,), bbox_inches='tight')



#%%
if __name__ == '__main__':

    # # Load smart meters information
    # sm_info_file = '..//Data//informations_households.csv'
    # df_sm_info = pd.read_csv(sm_info_file, header=0)

    # # Load weather data
    # weather_daily_file = '..//Data//weather_daily_darksky.csv'
    # df_weather = pd.read_csv(weather_daily_file, header=0)
    # df_weather['time'] = pd.to_datetime(df_weather['time'])
    # df_weather = df_weather.set_index('time')

    # # Load all data
    df_complete = read_acorn_group_blocks()

    # # Remove ToU SMs and split it into accorns
    # dict_data = split_into_acorns(df_complete, df_sm_info)

    # # Generate images
    # plot_std_or_tou_difference(df_sm_info, 
    #                            save_filename='Diff_Tarifas.png')

    # plot_total_energy_per_acorn(
    #     dict_data, 
    #     lst_acorn_names=['Affluent', 'Comfortable', 'Adversity', 'ACORN-U'],
    #     save_filename='Energia_Total_por_Acorn.png')

    # plot_total_relative_energy_per_acorn(
    #     dict_data, 
    #     lst_acorn_names=['Affluent', 'Comfortable','Adversity', 'ACORN-U'],
    #     save_filename='Energia_Relativa_Total_por_Acorn.png')
    
    # for weather_col in df_weather.columns:
    #     print(weather_col)
    #     if 'time' in weather_col.lower() or 'summary' in weather_col.lower():
    #         continue
    #     correlate_energy_weather_data(
    #         dict_data,
    #         lst_acorn_names=['Affluent', 'Comfortable', 'Adversity'],
    #         df_weather=df_weather,
    #         weather_col=weather_col,
    #         save_filename='Relação_Consumo_X_{}.png'.format(weather_col),
    #         figsize=(10,6)
    #     )

    #  # Load weather hourly data
    # weather_daily_file = '..//..//Data//weather_hourly_darksky.csv'
    # df_weather = pd.read_csv(weather_daily_file, header=0)
    # df_weather['time'] = pd.to_datetime(df_weather['time'])
    # df_weather = df_weather.set_index('time')
    
    # display(df_weather['temperature'].head())
    # rolling_size = 24*7
    # print('Rolling size: ', rolling_size/24, ' days.')
    # plt.plot(df_weather['temperature'].sort_index().rolling(rolling_size).mean()[::rolling_size])
    # plt.figure()
    # fft = plot_fourier_trsnfd_weather_data(df_weather['temperature'].sort_index().rolling(rolling_size).mean()[::rolling_size].dropna())
    
    # Load weather daily data
    weather_daily_file = '..//..//Data//weather_hourly_darksky.csv'
    desired_col = 'temperature'
    df_weather = pd.read_csv(weather_daily_file, header=0)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather = df_weather.set_index('time')
    thrs = 80
    
    display(df_weather[desired_col].head())
    rolling_size = 24*7
    weather_data = df_weather[desired_col].sort_index().rolling(rolling_size).\
                    mean()[::rolling_size].dropna()

    print('Rolling size: ', rolling_size, ' days.')
    # Plot weather signal FFT
    plt.figure(figsize=(10,6))
    fft_signal = plot_fourier_trsnfd_weather_data(weather_data.values,
                                                  rolling_size)
    xlim = np.arange(plt.gca().get_xlim()[0],plt.gca().get_xlim()[1])
    plt.plot(xlim, [thrs]*xlim.size)
    # plt.gca().set_ylim((0, 500))
    # Plot original data
    plt.figure(figsize=(10,6))
    plt.plot(weather_data)
    plt.plot(df_weather[desired_col].sort_index().rolling(rolling_size).mean(),
             color='indianred', alpha=.7)
    plt.plot(df_weather[desired_col].sort_index(), color='cornflowerblue', 
             alpha=.2)
    # Get signal with filtered frequencies
    fft_signal2 = fft_signal.copy()
    fft_signal2[np.abs(fft_signal) <= thrs] = 0
    plt.figure(figsize=(10,6))
    # plt.plot(ifft(ifftshift(fft_signal2)),color='cornflowerblue')
    plt.plot(
        ifft(ifftshift(np.where(np.abs(fft_signal)>150, fft_signal, 0))),
        color='mediumseagreen', label='150'
    )
    plt.plot(
        ifft(ifftshift(np.where(np.abs(fft_signal)>80, fft_signal, 0))),
        color='cornflowerblue', label='80'
    )
    plt.plot(
        ifft(ifftshift(np.where(np.abs(fft_signal)>50, fft_signal, 0))),
        color='indianred', label='50'
    )
    plt.gca().legend()

    # # Deal with residual temp data
    # residual = weather_data.values - ifft(ifftshift(fft_signal2))
    # plt.figure(figsize=(10,6))
    # residual_fft = plot_fourier_trsnfd_weather_data(residual,
    #                                                 rolling_size)
    # residual_fft2 = residual_fft.copy()
    # residual_fft2[np.abs(residual_fft) <= thrs] = 0
    # plt.figure(figsize=(10,6))
    # plt.plot(ifft(ifftshift(residual_fft2)),color='cornflowerblue')



    #TODO:  fazer uma função que mapeie a data de determinando sample com 
    #       a semana em que ele se encontra. Pois com a função desenvolvida
    #       hoje, já temos como obter a senoide que mais se aproxima em 
    #       questão de periodicidade da temperatura. E quando passamos os
    #       a semana correta, a função retorna o valor imediato também.
    
    #       Um detalhe é que a magnitude da função não está igual àquela 
    #       mostrada pelo sinal real, visto que filtramos varias outras
    #       frequências. Mas como vamos normalizar este dado, então não 
    #       teremos problema com isso.

    #       OBS.: Abandonou-se a ideia de pegar a média sem pular valores
    #       (window_step), o que é mostrado como a linha vermelha na segunda
    #       imagem, pois os resultados obtidos pela filtragem de frequências
    #       não foi nem um pouco satisfatório, quando comparado ao obtido
    #       pela linha azul. Isso acontece provavelmente porque ainda há
    #       um sinal muito ruidoso.


    # _, ax = plt.subplots(figsize=(10,8))
    # ax.plot(df_weather['temperature'].sort_index())

    # # Create senoid
    # temp = np.array(df_weather.index)

    # T_day = (temp[1] - temp[0]) * 24
    # f_day = 1. / float(T_day)

    # T_year = T_day * 365.2524
    # f_year = 1. / float(T_year)

    # sin_year = np.sin(2*np.pi*f_year * temp.astype(float))
    # cos_year = np.cos(2*np.pi*f_year * temp.astype(float))

    # ax.plot(sin_year, color='indianred')
    # ax.plot(cos_year, color='mediumseagreen')
