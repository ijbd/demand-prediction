import pandas as pd 
import numpy as np
from datetime import datetime
import argparse

FEATURE_COL_MAPPER = {"Demand (MW)" : "D",
						"Temperature (K)" : "T"}

def process_data(cleaned_data_file: str, 
				train_features_file: str, 
				train_labels_file: str,
				val_features_file: str, 
				val_labels_file: str,
				test_features_file: str,
				test_labels_file: str) -> None:
	'''
	Convert cleaned data into train/validation/test datasets for machine learning model. Save output to csv
	'''

	cleaned_data = load_cleaned_data(cleaned_data_file)

	# initialize dataframe
	processed_data = cleaned_data.copy()
	processed_data.rename(columns=FEATURE_COL_MAPPER, inplace=True)

	# Feature engineer fixed effects
	populate_fixed_effects(processed_data)
    
	# split into train/val/test
	train_data, val_data, test_data = split_data(processed_data)

	train_features, train_labels = split_features(train_data)
	val_features, val_labels = split_features(val_data)
	test_features, test_labels = split_features(test_data)

    # save 
	train_features.to_csv(train_features_file)
	train_labels.to_csv(train_labels_file)
	val_features.to_csv(val_features_file)
	val_labels.to_csv(val_labels_file)
	test_features.to_csv(test_features_file)
	test_labels.to_csv(test_labels_file)
	
	return None

def load_cleaned_data(cleaned_data_file: str) -> pd.DataFrame:

    cleaned_data = pd.read_csv(cleaned_data_file,index_col='Datetime',parse_dates=True)

    return cleaned_data

def populate_fixed_effects(processed_data: pd.DataFrame) -> None:
	'''
	Add fixed effect features to DataFrame:
	- weekday
	- trigonometric day of year
	'''
	
	processed_data['W'] = processed_data.index.weekday

	# calculate julian_day
	julian_day = processed_data.index.map(get_julian_day)

	processed_data['M-sine'] = np.sin(2 * np.pi * julian_day / 365.)
	processed_data['M-cosine'] = np.cos(2 * np.pi * julian_day / 365.)

	return None

def get_julian_day(date_time: datetime) -> int:

	return date_time.timetuple().tm_yday

def split_data(processed_data: pd.DataFrame) -> tuple:
	'''
	Split data into train, validation, and test dataset. 
	To include data spanning the available data range in each 
	subset while ensuring correct seasonal distributions,
	months are shuffled between years creating a 50/25/25 split.
	'''

	# initialize dataset
	test_data = pd.DataFrame()
	val_data = pd.DataFrame()

	# create shuffled months
	np.random.seed(1)

	# find training size
	years = np.unique(processed_data.index.year)

	num_training_years = len(years) // 4

	# shuffle years for training and validation data
	test_val_years = [np.random.choice(range(min(years),max(years)+1),2*num_training_years,replace=False) for i in range(12)]

	for month in range(12):
		
		data_in_month = processed_data[processed_data.index.month == month+1]

		test_in_month = data_in_month[data_in_month.index.year.isin(test_val_years[month][:num_training_years])]
		test_data = pd.concat((test_data, test_in_month))

		val_in_month = data_in_month[data_in_month.index.year.isin(test_val_years[month][num_training_years:])]
		val_data = pd.concat((val_data, val_in_month))
		
	train_data = processed_data[~processed_data.index.isin(test_data.index.append(val_data.index))]

	return train_data, val_data, test_data

def split_features(dataset: pd.DataFrame) -> tuple:

	features = dataset.copy()
	labels = features.pop('D')

	return features, labels

if __name__ == '__main__':
	# argument parsing
	parser = argparse.ArgumentParser('process_data',description='Processed cleaned dataset into features and labels for model.')
	parser.add_argument('cleaned_data_file',type=str)
	parser.add_argument('train_features_file',type=str)
	parser.add_argument('train_labels_file',type=str)
	parser.add_argument('val_features_file',type=str)
	parser.add_argument('val_labels_file',type=str)
	parser.add_argument('test_features_file',type=str)
	parser.add_argument('test_labels_file',type=str)
	args = parser.parse_args()
	
	# data cleaning
	process_data(args.cleaned_data_file,
					args.train_features_file,
					args.train_labels_file,
					args.val_features_file,
					args.val_labels_file,
					args.test_features_file,
					args.test_labels_file)

