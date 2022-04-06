import pandas as pd
from datetime import timedelta

import argparse

DEMAND_COL_MAPPER = { "cleaned demand (MW)" : "Demand (MW)"}
TEMP_COL_MAPPER = {"Temperature (K)": "Temperature (K)"}

def clean_data(raw_demand_filepath: str, 
                raw_temp_filepath: str,
                cleaned_data_filepath: str, 
                years: list) -> None:
    """
    Clean and combine the raw temperature and demand data.
    """
    
    # open raw data
    raw_demand = load_raw_demand(raw_demand_filepath)
    raw_temp = load_raw_temp(raw_temp_filepath)

    # clean and format dataframes
    cleaned_demand = format_dataframe(raw_demand, DEMAND_COL_MAPPER, years)
    cleaned_temp = format_dataframe(raw_temp, TEMP_COL_MAPPER, years)

    # combine data
    cleaned_data = pd.concat((cleaned_temp,cleaned_demand),axis=1)
    
    # save data cleaned_data
    cleaned_data.to_csv(cleaned_data_filepath)

    return None

def load_raw_demand(raw_demand_filepath: str) -> pd.DataFrame:

	raw_demand = pd.read_csv(raw_demand_filepath,
							parse_dates=True,
							index_col='date_time',
							usecols=['date_time','cleaned demand (MW)'])

	return raw_demand

def load_raw_temp(raw_temp_filepath: str) -> pd.DataFrame:

	raw_temp = pd.read_csv(raw_temp_filepath,
                            index_col=0,
                            parse_dates=True)

	return raw_temp


def format_dataframe(raw_data:pd.DataFrame, col_mapper: dict, years:list) -> pd.DataFrame:

    # make new dataframe
    cleaned_data = raw_data.copy()

    # rename columns
    cleaned_data.index.rename("Datetime", inplace=True)
    cleaned_data.rename(columns=col_mapper, inplace=True)

    # resample to daily
    cleaned_data = cleaned_data.resample(timedelta(days=1)).max()
    
    # remove leap day
    cleaned_data = cleaned_data[~((cleaned_data.index.month == 2) & (cleaned_data.index.day == 29))]

    # filter by years
    cleaned_data = cleaned_data.loc[cleaned_data.index.year.isin(years)]

    return cleaned_data

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser('clean_data',description='Formats and combines raw demand and temp data.')
    parser.add_argument('raw_demand_filepath',type=str)
    parser.add_argument('raw_temp_filepath',type=str)
    parser.add_argument('cleaned_data_filepath',type=str)
    parser.add_argument('years', nargs='+', type=int, help="data years")
    
    args = parser.parse_args()

    # data cleaning
    clean_data(args.raw_demand_filepath, 
                args.raw_temp_filepath,
                args.cleaned_data_filepath,
                args.years)