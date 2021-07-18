#%% 
# Imports 
import pandas as pd
from utils.exploratory_data_analysis import (read_acorn_group_blocks,
                                             split_into_acorns,
                                             plot_std_or_tou_difference,
                                             plot_total_energy_per_acorn,
                                             plot_total_relative_energy_per_acorn,
                                             correlate_energy_weather_data,
                                             plot_fourier_trsnfd_weather_data) 

#%%
if __name__ == '__main_':

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

if __name__ == '__main__':
    hourly_data_file = '..//Data//weather_hourly_darksky.csv'
    df_weather = pd.read_csv(hourly_data_file, usecols=['time','temperature'])

    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather = df_weather.set_index('time')['temperature']
    
    display(df_weather.head())

    fft = plot_fourier_trsnfd_weather_data(df_weather)