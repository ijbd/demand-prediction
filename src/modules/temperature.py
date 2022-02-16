import pandas as pd
import os

from datetime import datetime, timedelta

def load_raw_temperature(raw_temperature_filepath):
	'''
	Load hourly raw temperature CSV into dataframe. 

	:param raw_temperature_filepath: Filepath.
	:type raw_temperature_filepath: str
	:returns: Dataframe object with temperature, date, and time variables
	:rtype: pd.Dataframe
	'''
	
	assert os.path.exists(raw_temperature_filepath), "Missing raw temperature data"

	raw_temperature = pd.read_csv(raw_temperature_filepath,
									skiprows=2,
									usecols=['Year','Month','Day','Hour','Temperature'])

	return raw_temperature
			
def clean_raw_temperature(raw_temperature, year_from, year_to):
	'''
	Reformat raw temperature dataframe to match interface for range [year_from,year_to]:
	- rename columns
	- resample hourly values to daily maximum
	- remove leap days
	- filter by year

	:param raw_temperature: Raw temperature datafrmae from load_raw_temperature().
	:type raw_temperature: pd.Dataframe
	:param year_from: First year to include.
	:type year_from: int
	:param year_to: Last year to include (inclusive).
	:type year_to: int
	:returns: Cleaned temperature dataframe.
	:rtype: pd.Dataframe
	'''

	# reindex
	cleaned_temperature = convert_datetime_to_index(raw_temperature)

	# rename
	cleaned_temperature.index.rename('Datetime',inplace=True)
	cleaned_temperature.rename(columns={'Temperature' : 'Temperature (C)'}, inplace=True)

	# resample to daily values
	cleaned_temperature = cleaned_temperature.resample(timedelta(days=1)).max()

	# remove leap days 
	cleaned_temperature = cleaned_temperature[~((cleaned_temperature.index.month == 2) &
											 	(cleaned_temperature.index.day == 29))]

	# filter years
	cleaned_temperature = cleaned_temperature.loc[datetime(year_from,1,1):datetime(year_to,12,31)]

	return cleaned_temperature
	
def convert_datetime_to_index(raw_temperature):
	'''
	Return new dataframe converting date and time variables into datetime index

	:param raw_temperature: Dataframe with date and time columns ('Year', 'Month', 'Day', 'Hour').
	:type raw_temperature: pd.DataFrame
	:returns: New dataframe with datetime index and date time variable columsn removed.
	:rtype: pd.DataFrame
	'''
	reindexed_temperature = raw_temperature.copy()

	# reindex into date time objects
	reindexed_temperature['date_time'] = pd.to_datetime(reindexed_temperature[['Year','Month','Day','Hour']])        
	reindexed_temperature.set_index('date_time',drop=True,inplace=True)
	
	# drop date time dummy columns
	reindexed_temperature.drop(columns=['Year','Month','Day','Hour'],inplace=True)

	return reindexed_temperature
