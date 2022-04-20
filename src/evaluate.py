import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

def evaluate(config: dict) -> None:

	# setup 
	summary = pd.read_csv(config["ann_summary_file"], squeeze=True)

	# load model
	model = tf.keras.models.load_model(config["ann_model_file"])

	# iterate train/val/test 
	for dataset in ["train", "val", "test"]:

		# load features/labels
		features = pd.read_csv(config[f"{dataset}_features_file"], index_col='Datetime', parse_dates=True)
		labels = pd.read_csv(config[f"{dataset}_labels_file"], index_col='Datetime', parse_dates=True)
		
		# make predictions
		predictions = model.predict(features)

		# get metrics
		summary[f"{dataset}-rmse"] = mean_squared_error(labels, predictions,squared=False)
		summary[f"{dataset}-mape"] = mean_absolute_percentage_error(labels, predictions)
		summary[f"{dataset}-r2"] = r2_score(labels, predictions)

		# get peak hours 
		peak_hours = labels > np.percentile(labels,75)
		summary[f"{dataset}-rmse-25"] = mean_squared_error(labels[peak_hours], predictions[peak_hours],squared=False)
		summary[f"{dataset}-mape-25"] = mean_absolute_percentage_error(labels[peak_hours], predictions[peak_hours])
		summary[f"{dataset}-r2-25"] = r2_score(labels[peak_hours], predictions[peak_hours])
	
	# write summary to csv
	summary.to_csv(config["ann_summary_file"], header=False)
	
	# write predictions to csv
	predictions = pd.Series(index=labels.index, data=predictions)
	predictions.to_csv(config["ann_test_predictions_file"])
	return None

