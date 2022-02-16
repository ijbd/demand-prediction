import pandas as pd 
import numpy as np
import tensorflow as tf
import argparse

from clean_data import load_cleaned_data

def process_data(cleaned_data_filepath,processed_data_filepath):
	'''
	TODO
	'''

	cleaned_data = load_cleaned_data(cleaned_data_filepath)

	# initialize dataframe
	processed_data = pd.DataFrame()

	# populate
	populate_demand_and_temperature(processed_data,cleaned_data)
	populate_fixed_effects(processed_data)
    
    # save 
	processed_data.to_csv(processed_data_filepath)

def populate_demand_and_temperature(processed_data,cleaned_data):
	'''
	Copy demand and temperature data from cleaned dataset into processed dataset.

	:param processed_data: DataFrame to which data is copied (CHANGED).
	:type processed_data: pd.DataFrame
	:param cleaned_data: DataFrame from which data is copied.
	:type cleaned_data: pd.DataFrame
	:returns: None
	:rtype: None
	'''

	processed_data['T'] = cleaned_data['Temperature (C)']
	processed_data['D'] = cleaned_data['Demand (MW)']

	return None

def populate_fixed_effects(processed_data):
	'''
	Add fixed effect features to DataFrame:
	- weekday
	- trigonometric day of year

	:param processed_data: TODO
	:type processed_data: TODO
	:returns: None
	:rtype: None
	'''
	
	processed_data['W'] = processed_data.index.weekday

	# calculate julian_day
	julian_day = processed_data.index.map(get_julian_day)

	processed_data['M-sine'] = np.sin(2 * np.pi * julian_day / 365.)
	processed_data['M-cosine'] = np.cos(2 * np.pi * julian_day / 365.)

	return None

def get_julian_day(date_time):
	'''
	Return julian day of a datetime object.

	:param date_time: Single date time object,
	:type date_time: TODO
	:returns: Julian day.
	:rtype: Int
	'''
	return date_time.timetuple().tm_yday

def load_processed_data(processed_data_filepath):
    '''
	TODO
	'''

    # load 
    processed_data = pd.read_csv(processed_data_filepath,index_col='Datetime',parse_dates=True)

    return processed_data

if __name__ == '__main__':
	# argument parsing
	parser = argparse.ArgumentParser('process_data',description='Processed cleaned dataset into features and labels for model.')
	parser.add_argument('cleaned_data_filepath',type=str)
	parser.add_argument('processed_data_filepath',type=str)
	args = parser.parse_args()
	
	# data cleaning
	process_data(args.cleaned_data_filepath,args.processed_data_filepath)

