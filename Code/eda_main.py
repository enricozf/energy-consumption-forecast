#%% 
# Imports 
import pandas as pd
from utils.exploratory_data_analysis import (read_acorn_group_blocks,
                                             split_into_acorns,
                                             plot_std_or_tou_difference,
                                             plot_total_energy_per_acorn,
                                             plot_total_relative_energy_per_acorn,
                                             correlate_energy_weather_data,
                                             plot_fourier_trsnfd_weather_data,
                                             represent_temperature_fft,
                                             scatter_temperature_consumption,
                                             show_week_day_consumption_diff,
                                             insert_representation_in_csv) 

def save_energy_group_figs():
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

def save_energy_weather_figs():
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

#%%
if __name__ == '__main__':
    
    data_path = '..//Data//daily_dataset.csv//daily_dataset.csv'
    info_lclid_path = '..//Data//informations_households.csv'
    result = 0.15845344960689545
    df = pd.read_csv(data_path)
    df_info = pd.read_csv(info_lclid_path)
    mask = (df_info['stdorToU'] == 'Std') & (df_info['Acorn_grouped'] == 'Affluent')
    affluent_nonToU_lclids = list(df_info.loc[mask, 'LCLid'])
    df_filt = df[df['energy_count']==48]
    print("Número inicial de medicoes com 48: ", df_filt.shape[0])
    df_filt = df_filt.set_index('LCLid').loc[affluent_nonToU_lclids,:]
    print("Número inicial de medicoes com 48: ", df_filt.shape[0])
    print('Consumo médio por dia é: ', df_filt['energy_sum'].mean())
    print('Consumo médio em 5hrs é: ', df_filt['energy_sum'].mean()/48*10)
    print('Consumo médio em 5hrs é: ', result**0.5/(df_filt['energy_sum'].mean()/48*10))

    # insert_representation_in_csv(desired_col='temperature')
    # data_path = '..//Data//halfhourly_dataset//block_0.csv'
    # df_block_hh = pd.read_csv(data_path, header=0)

    # data_path = '..//Data//daily_dataset//block_0.csv'
    # df_block_day = pd.read_csv(data_path, header=0)
    
    # data_path = '..//Data//uk_bank_holidays.csv'
    # df_holiday = pd.read_csv(data_path, header=0)
    
    # data_path = '..//Data//weather_hourly_darksky.csv'
    # df_weather_h = pd.read_csv(data_path, header=0)
    
    # data_path = '..//Data//weather_DAILY_darksky.csv'
    # df_weather_day = pd.read_csv(data_path, header=0)
# %%
