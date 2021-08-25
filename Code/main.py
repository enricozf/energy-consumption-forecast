#%%
import pandas as pd
from utils.exploratory_data_analysis import (read_acorn_group_blocks, 
                                             split_into_acorns)

def create_df(
    consumption_file_path : str ='..//Data//halfhourly_dataset',
    sm_info_file : str = '..//Data//informations_households.csv',
    lst_acorn_names : str = ['Affluent', 'Comfortable', 'Adversity'],
    blocks_per_acorn : dict = {
        'Affluent' : (0, 43),
        'Comfortable' : (43, 73),
        'Adversity' : (74, 110)},
    day_col : str = 'tstp',
    save_filename : str =None,
    save_path : str = '..//Figures//',
    fontsize=10):
    
    # Read all Smart meters data
    dict_data = read_acorn_group_blocks(consumption_file_path)
    # Load smart meters information
    df_sm_info = pd.read_csv(sm_info_file, header=0)
    # Split into acorns
    dict_data = split_into_acorns(dict_data, df_sm_info)

    return dict_data

def monday_or_weekend(elem):
    return 1 if elem.weekday() in [0,5,6] else 0

if __name__ == '__main__':
    block_file_name = '..//Data//halfhourly_dataset//block_0.csv'
    weather_file_name = '..//Data//weather_hourly_darksky_final.csv'
    time_col_data = 'tstp'
    time_col_weather = 'time'
    desired_weather_cols = ['apparentTemperature', 'weekly_temperature'] #uvIndex

    # TODO: Merge + fill NaN
    df = pd.read_csv(block_file_name, header=0)
    df[time_col_data] = pd.to_datetime(df[time_col_data])
    df['MondayORWE'] = df[time_col_data].apply(monday_or_weekend)
    # df['weekday'] = df[time_col_data].apply(lambda x: x.weekday())
    df.set_index(time_col_data, inplace=True)

    # Create weather df
    weather_df = pd.read_csv(weather_file_name, header=0)
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df.set_index(time_col_weather, inplace=True)

    # Include weather features in main DataFrame
    for w_col in desired_weather_cols:
        weather_sr = pd.Series(
            data=weather_df[w_col].rolling(2).mean()[1:].values,
            index=(weather_df.index[:-1]+pd.Timedelta(30, unit='min')).values
        )
        weather_sr = pd.concat([weather_df[w_col], weather_sr]).sort_index()
        # Insert weather data to main data
        df = pd.merge(df, 
                      pd.DataFrame(weather_sr).rename(columns={0:w_col}),
                      how='inner', left_index=True, right_index=True)


