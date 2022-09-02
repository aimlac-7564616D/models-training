import pandas as pd
import numpy as np


def frame_stats(frame:pd.DataFrame) -> float:
    """ Displays the timeline statistics.

    Calculates the minimun, maximum and avarage difference between successive timestamps in the dataframe

    Args:
        frame (pd.DataFrame): Any DataFrame with a time column

    Returns: 
        min, max, mean 
    """
    diff = (frame['time'].diff()).dt.seconds 
    return np.min(diff), np.max(diff), np.mean(diff)

def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    """ Extracts the longest continuous time series.

    Take a dataframe and converts all timestamps to the closest 30mins, then extracts the longest continuous time series (gaps less than 1hr between 
    datapoints )

    Args:
        frame (pd.DataFrame): Any DataFrame with a time column

    Returns: 
        pd.DataFrame: A dataframe with gaps no larger than 1hr
    """
    # convert time column to datetime
    frame['time'] = pd.to_datetime(frame['time'])

    # round all time to the closest half an hour
    frame['time'] = frame['time'].round('30min')

    frame = frame.drop_duplicates(subset=['time'])

    diff = (frame['time'].diff()).dt.seconds 

    inds = np.array([0])
    gap_inds = np.where(diff > (1*(60**2)))[0] 
    inds = np.append(inds, gap_inds)
    inds = np.append(inds, np.array(len(diff) - 1))

    # find the longest timeframe
    longest = 0
    s, e = 0, 0
    for i in range(len(inds) - 1):
        no_points = inds[i+1] - inds[i]
        if no_points > longest:
            longest = no_points
            s, e = inds[i], inds[i+1]

    return frame.iloc[s:e, :]

def insert_synthetic_data(frame:pd.DataFrame) -> pd.DataFrame:
    """Inserts synthetic data into the dataframe

    Ensures that the dataframe is in 30min intervals and inputs synthetic data for any timestamp that was not already 
    present in the data

    Args:
        frame (pd.DataFrame): The dataframe you want to add data too

    Returns: 
        pd.DataFrame: A dataframe with 30min intervals
    """

    n = len(frame['time'])

    # generate the new times - every 30mins
    date_range = pd.date_range(frame['time'].iloc[0], periods=n*2, freq=".5H")

    # create array of NaN for all the times
    null_data = np.empty(shape=(len(date_range), frame.shape[1] - 1))
    null_data[:] = np.NaN

    # create a dataframe with the null data
    new_times = pd.DataFrame(date_range)
    null_data = pd.DataFrame(null_data)

    new_data = pd.concat([new_times, null_data], axis=1, ignore_index=True)    
    
    # Store the column names of the dataframe
    col_names = {}
    for i in range(len(frame.columns)):
        col_names[i] = frame.columns[i]

    new_data = new_data.rename(columns=col_names)
    
    toReutrn = pd.concat([frame, new_data], axis=0)
    toReturn = toReutrn.drop_duplicates(subset=['time'])

    # sort the times and reset the index of the dataframe
    toReturn = toReturn.sort_values(by=['time'])

    # interpolate (issues) the data - using ffill and bfill instead
    toReturn.iloc[:, 1:] = toReturn.iloc[:,1:].interpolate(method='linear', axis=0)
    toReturn = toReturn.ffill()

    return toReturn

def synchronise_data(frame1:pd.DataFrame, frame2:pd.DataFrame, frame3:pd.DataFrame) -> pd.DataFrame:

    """Synchronises the time series of met, open, and energy dataframes

    Makes sure that all 3 input dataframes begin and end at the same times

    Args: 
        frame1 (pd.DataFrame): un-synchronised dataframe
        frame2 (pd.DataFrame): un-synchronised dataframe
        frame3 (pd.DataFrame): un-synchronised dataframe

    Returns:
        frame1 (pd.DataFrame): Synchronised dataframe
        frame2 (pd.DataFrame): Synchronised dataframe
        frame3 (pd.DataFrame): Synchronised dataframe
    """

    s = [frame1['time'].iloc[0], frame2['time'].iloc[0], frame3['time'].iloc[0]]
    e = [frame1['time'].iloc[-1], frame2['time'].iloc[-1], frame3['time'].iloc[-1]]

    frame1.set_index('time', inplace=True)
    frame2.set_index('time', inplace=True)
    frame3.set_index('time', inplace=True)

    start, end = np.max(s), np.min(e)

    return frame1.loc[start:end], frame2.loc[start:end], frame3.loc[start:end]

def generate_wind_data(met:pd.DataFrame, open:pd.DataFrame, energy:pd.DataFrame) -> pd.DataFrame:
    
    """Generates the data required for the wind model

    Extracts the valuable columns from each of the input dataframes and the target variable required for training 

    Args: 
        met (pd.DataFrame) : MetOffice synchronised dataframe
        open (pd.DataFrame) : OpenWeather synchronised dataframe
        energy (pd.DataFrame) : EnergyOnsite synchronised dataframe

    Returns:
        pd.DataFrame : dataframe containing the driving series of the network along with the target variable
    """

    met_cols = ['windDirectionFrom10m', 'windGustSpeed10m', 'max10mWindGust', 'windSpeed10m']
    open_cols = ['wind_deg', 'wind_speed', 'wind_gust']
    target_col = ['wind1']

    print(energy.columns)

    wind_data = pd.concat([met[met_cols], open[open_cols]], axis=1)
    wind_data = pd.concat([wind_data, energy[target_col]], axis=1)

    return wind_data

def generate_solar_data(met:pd.DataFrame, open:pd.DataFrame, energy:pd.DataFrame) -> pd.DataFrame:

    """Generates the data required for the solar model

    Extracts the valuable columns from each of the input dataframes and the target variable required for training 

    Args: 
        met (pd.DataFrame) : MetOffice synchronised dataframe
        open (pd.DataFrame) : OpenWeather synchronised dataframe
        energy (pd.DataFrame) : EnergyOnsite synchronised dataframe

    Returns:
        pd.DataFrame : dataframe containing the driving series of the network along with the target variable
    """
    
    met_cols = ['visibility', 'uvIndex', 'significantWeatherCode', 'probOfPrecipitation', 'screenTemperature', 'feelsLikeTemperature']
    open_cols = ['visibility', 'clouds', 'temperature']
    target_col = ['solar']

    solar_data = pd.concat([met[met_cols], open[open_cols]], axis=1)
    solar_data = pd.concat([solar_data, energy[target_col]], axis=1)

    return solar_data


def extract_data_from_daterange(df:pd.DataFrame, start_date, end_date):
    return df.loc[start_date:end_date, :]
    




