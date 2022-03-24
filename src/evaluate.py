import numpy as np
import pandas as pd
import argparse
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from tensorflow import keras


from process_data import load_processed_data
from modules.model import split_data, split_features, train_model, get_normalizer
from hyperparameter_search import get_best_model, get_tuner, build_model_for_search

def calculate_metrics(test_labels, test_predictions):
	metrics = dict()

	metrics['rmse'] = mean_squared_error(test_labels, test_predictions,squared=False)
	metrics['mape'] = mean_absolute_percentage_error(test_labels, test_predictions)
	metrics['r2'] = r2_score(test_labels, test_predictions)

	# peak metrics 
	peak_hours = test_labels > np.percentile(test_labels,75)
	metrics['rmse-25'] = mean_squared_error(test_labels[peak_hours], test_predictions[peak_hours],squared=False)
	metrics['mape-25'] = mean_absolute_percentage_error(test_labels[peak_hours], test_predictions[peak_hours])
	metrics['r2-25'] = r2_score(test_labels[peak_hours], test_predictions[peak_hours])

	return metrics

def print_metrics(metrics):
	for m in metrics:
		print(f"{m:10s}:\t{metrics[m]:3.3f}")

def evaluate_model(processed_data_filepath, search_dir, search_name, results_filepath):
	
	# get data
	processed_data = load_processed_data(processed_data_filepath)

	train_dataset, val_dataset, test_dataset = split_data(processed_data)

	train_features, train_labels = split_features(train_dataset)
	val_features, val_labels = split_features(val_dataset)
	test_features, test_labels = split_features(test_dataset)

	# get tuner
	normalizer = get_normalizer(train_features)
	build_model_for_search.normalizer = normalizer
	tuner = get_tuner(search_dir, search_name)

	# get model
	model = get_best_model(tuner)
	
	#  calculate results
	test_predictions = model.predict(test_features)
	metrics = calculate_metrics(test_labels, test_predictions)

	# open results
	print_metrics(metrics)

if __name__ == '__main__':
		# argument parsing
	parser = argparse.ArgumentParser('evaluate',description='Evaluate model for .')
	parser.add_argument('processed_data_filepath',type=str)
	parser.add_argument('search_dir',type=str)
	parser.add_argument('search_name',type=str)
	parser.add_argument('results_filepath',type=str)
	args = parser.parse_args()
	
	# evaluate
	evaluate_model(args.processed_data_filepath,args.search_dir,args.search_name,args.results_filepath)

