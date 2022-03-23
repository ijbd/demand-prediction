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
									index_col=0,
									parse_dates=True)

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

	# rename
	raw_temperature.index.rename('Datetime', inplace=True)

	# resample to daily values
	cleaned_temperature = raw_temperature.resample(timedelta(days=1)).max()

	# filter years
	cleaned_temperature = cleaned_temperature.loc[datetime(year_from,1,1):datetime(year_to,12,31)]

	return cleaned_temperature
	