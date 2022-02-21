from multiprocessing import ProcessError
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import argparse
import pickle

from process_data import load_processed_data
from modules.model import build_model, split_data, split_features, get_normalizer

def build_model_for_search(hyperparameters):
	'''
	Build model using keras tuner hyperparameters object.
	This function also defines the hyperparameter search space
	*Note that build_model_for_search.normalizer must be defined ahead of time
	
	:param hyperparameters: Hyperparameters object with model architecture.
	:type hyperparameters: keras_tuner.engine.hyperparameters.HyperParameters
	:returns: Compiled neural network (using build_model() from model.py)
	:rtype: 
	'''

	# define solution space for fixed-length hyperparameters
	min_hidden_layers = 1
	max_hidden_layers = 5
	hidden_layers = hyperparameters.Int("hiddenLayers",min_hidden_layers,max_hidden_layers)
	learning_rate = hyperparameters.Float("lr", min_value=5e-4, max_value=1e-1, sampling="log")
	
	# define solution space for variable-length hyperparameters
	units = []
	dropout = []

	for i in range(5):
		units.append(hyperparameters.Choice(f"units_{i}", [64,128,256,512,1028]))
		dropout.append(hyperparameters.Choice(f"dropout_{i}", [.0, .1, .2]))
	
	return build_model(build_model_for_search.normalizer,hidden_layers,units,dropout,learning_rate)

def hyperparameter_search(processed_data_filepath,search_dir,search_name,model_filepath):
	'''
	Use keras tuner to find optimal model architecture.

	:param cleaned_data_filepath: Filepath of processed data CSV.
    :type cleaned_data_filepath: str
	:param search_dir: Filepath to store results.
	:type search_dir: str
	:param search_name: Label for hyperparameter search.
	:type search_name: str
	:param model_filepath: Location to save best model.
	:type model_filepath: str
	:returns: None
	:rtype: None
	'''
	# get data
	processed_data = load_processed_data(processed_data_filepath)

	train_dataset, val_dataset, test_dataset = split_data(processed_data)

	train_features, train_labels = split_features(train_dataset)
	val_features, val_labels = split_features(val_dataset)
	test_features, test_labels = split_features(test_dataset)

	# assign normalizer
	build_model_for_search.normalizer = get_normalizer(train_features)

	# build keras tuner
	tuner = get_tuner(search_dir, search_name)

	# search
	search(tuner,train_features,train_labels,val_features,val_labels)

	return None
	
def get_tuner(search_dir,search_name):
	'''
	Create or load an instance of a keras hyperparameter tuning object.
	For a given tuner, no overwrite will occur (even with changed parameters).

	:param search_dir: Directory to store results.
	:type search_dir: str
	:param search_name: Name of search instance.
	:type search_dir: str
	:returns: Keras tuner object.
	:rtype: keras_tuner.tuners
	''' 
	# define search parameters
	tuner = kt.RandomSearch(
		build_model_for_search,
		objective='val_loss',
		max_trials=200,
		directory=search_dir,
		project_name=search_name
	)

	return tuner

def get_best_hyperparameters(tuner):
	return tuner.get_best_hyperparameters()[0]

def get_best_model(tuner):
	return tuner.get_best_models()[0]

def search(tuner,train_features,train_labels,val_features,val_labels):
	'''
	Carry out search for Keras tuner instance. 

	:param tuner: Tuner from get_tuner()
	:type tuner: keras_tuner.tuners
	:param train_features: Train features compatible with tuner model architecture. 
	:type train_features: pd.DataFrame
	:param train_labels: Train labels compatible with tuner model architecture. 
	:type train_labels: pd.DataFrame
	:param val_features: Validation features compatible with tuner model architecture. 
	:type val_features: pd.DataFrame
	:param val_labels: Validation labels compatible with tuner model architecture. 
	:type val_labels: pd.DataFrame
	:returns: None
	:rtype: None
	'''
	# add early stopping 
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

	# search
	tuner.search(train_features, 
			train_labels,
			epochs=5000,
			validation_data=(val_features, val_labels),
			verbose=False,
			callbacks=[early_stop])

	return None

if __name__ == '__main__':
	# argument parsing
	parser = argparse.ArgumentParser('hyperparameter search',description='Find best ANN architecture for dataset.')
	parser.add_argument('processed_data_filepath',type=str)
	parser.add_argument('search_dir',type=str)
	parser.add_argument('bal_authority',type=str)
	parser.add_argument('model_filepath',type=str)
	args = parser.parse_args()
	
	# search
	hyperparameter_search(args.processed_data_filepath,
						args.search_dir,
						args.bal_authority,
						args.model_filepath)

