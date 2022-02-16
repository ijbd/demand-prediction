import pandas as pd
import os
from datetime import datetime, timedelta

def load_raw_demand(raw_demand_filepath):
	'''
	Load hourly raw demand from CSV into series. See download_demand.py for source.

	:param raw_demand_filepath: Filepath.
	:type raw_demand_filepath: str
	:returns: Series object with datetime index and demand column.
	:rtype: pd.Series
	'''
	raw_demand = pd.read_csv(raw_demand_filepath,
							parse_dates=True,
							index_col='date_time',
							usecols=['date_time','cleaned demand (MW)'])

	return raw_demand

def clean_raw_demand(raw_demand, year_from, year_to):
	'''
	Reformat demand dataframe to match interface for time range [year_from,year_to]:
	- rename columns
	- resample hourly values to daily maximum
	- remove leap days
	- filter by year

	:param raw_demand: Raw demand series from load_raw_demand().
	:type raw_demand: pd.Series
	:param year_from: First year to include.
	:type year_from: int
	:param year_to: Last year to include (inclusive).
	:type year_to: int
	:returns: Cleaned demand dataframe.
	:rtype: pd.Dataframe
	'''
	
	cleaned_demand = raw_demand.copy()

	# rename
	cleaned_demand.index.rename('Datetime',inplace=True)
	cleaned_demand.rename(columns={'cleaned demand (MW)' : 'Demand (MW)'}, inplace=True)

	# resample to daily maximum values
	cleaned_demand = cleaned_demand.resample(timedelta(days=1)).max()

	# remove leap days 
	cleaned_demand = cleaned_demand[~((cleaned_demand.index.month == 2) & 
										(cleaned_demand.index.day == 29))]

	# filter years
	cleaned_demand = cleaned_demand.loc[datetime(year_from,1,1):datetime(year_to,12,31)]

	return cleaned_demand