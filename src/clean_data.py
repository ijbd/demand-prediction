import pandas as pd
from datetime import timedelta, datetime
import os, sys

import argparse

from modules.demand import load_raw_demand, clean_raw_demand
from modules.temperature import load_raw_temperature, clean_raw_temperature

def clean_data(raw_demand_filepath, 
                raw_temperature_filepath,
                cleaned_data_filepath, 
                year_from, 
                year_to):
    '''
	Clean, combine, then save raw demand and temperature data for time range [year_from, year_to].

    :param raw_demand_filepath: Filepath to raw demand CSV.
    :type raw_demand_filepath: str
    :param raw_temperature_filepath: Filepath to raw temperature CSV.
    :type raw_temperature_filepath: str
    :param cleaned_data_filepath: Filepath to save cleaned data CSV.
    :type cleaned_data_filepath: str
    :param year_from: First year of data to include.
    :type year_from: int
    :param year_to: Last year of data to include.
    :type year_to: int
    :returns: None
    :rtype: None
	'''

    # open raw data
    raw_demand = load_raw_demand(raw_demand_filepath)
    cleaned_demand = clean_raw_demand(raw_demand, year_from, year_to)

    # open temperature data
    raw_temperature = load_raw_temperature(raw_temperature_filepath)
    cleaned_temperature = clean_raw_temperature(raw_temperature, year_from, year_to)

    # combine data
    cleaned_data = pd.concat((cleaned_temperature,cleaned_demand),axis=1)
    
    # save data cleaned_data
    cleaned_data.to_csv(cleaned_data_filepath)

    return None


def load_cleaned_data(cleaned_data_filepath):
    '''
	Loads cleaned demand data from CSV. Cleaned data should be generated with clean_data().

    :param cleaned_data_filepath: Filepath of cleaned data CSV.
    :type cleaned_data_filepath: str
    :returns: Dataframe with daily peak demand and temperature (indexed by datetime).
    :rtype: pd.DataFrame
	'''

    cleaned_data = pd.read_csv(cleaned_data_filepath,index_col='Datetime',parse_dates=True)

    return cleaned_data
    
if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser('clean_data',description='Formats and combines raw demand and temperature data.')
    parser.add_argument('raw_demand_filepath',type=str)
    parser.add_argument('raw_temperature_filepath',type=str)
    parser.add_argument('cleaned_data_filepath',type=str)
    parser.add_argument('year_from',type=int,choices=range(2016,2020),default=2016)
    parser.add_argument('year_to',type=int,choices=range(2016,2020),default=2019)
    args = parser.parse_args()

    # data cleaning
    clean_data(args.raw_demand_filepath, 
                args.raw_temperature_filepath,
                args.cleaned_data_filepath,
                args.year_from, 
                args.year_to)