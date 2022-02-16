import numpy as np
import pandas as pd
import argparse
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from tensorflow import keras


from process_data import load_processed_data
from modules.model import split_data, split_features, get_normalizer, train_model

def evaluate_model(processed_data_filepath, model_filepath, bal_authority, results_filepath):
	'''
	TODO
	'''
	# get data
	processed_data = load_processed_data(processed_data_filepath)

	train_dataset, val_dataset, test_dataset = split_data(processed_data)

	train_features, train_labels = split_features(train_dataset)
	val_features, val_labels = split_features(val_dataset)
	test_features, test_labels = split_features(test_dataset)

	# get model
	model = keras.models.load_model(model_filepath)

	# train model
	#train_model(model, train_features, train_labels, val_features, val_labels)
	
	#  calculate results
	test_predictions = model.predict(test_features)
	metrics = calculate_metrics(test_labels, test_predictions)

	# open results
	results = load_results(results_filepath)

	# append result
	results = append_results(bal_authority, metrics, results)

	# save results
	results.to_csv(results_filepath,index_label='bal_authority')

def append_results(bal_authority, metrics, results):
	new_row = pd.DataFrame(metrics,index=[bal_authority])

	if bal_authority in results.index:
		results.loc[bal_authority] = new_row.loc[bal_authority]
	else:
		results = pd.concat([results,new_row])
	return results

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

def load_results(results_filepath):
	'''
	TODO
	'''
	if os.path.exists(results_filepath):
		results = pd.read_csv(results_filepath,index_col=['bal_authority'])
	else:
		results = pd.DataFrame()

	return results


	
if __name__ == '__main__':
		# argument parsing
	parser = argparse.ArgumentParser('evaluate',description='Evaluate model for .')
	parser.add_argument('processed_data_filepath',type=str)
	parser.add_argument('model_filepath',type=str)
	parser.add_argument('bal_authority',type=str)
	parser.add_argument('results_filepath',type=str)
	args = parser.parse_args()
	
	# evaluate
	evaluate_model(args.processed_data_filepath,args.model_filepath,args.bal_authority,args.results_filepath)

