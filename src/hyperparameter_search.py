import os
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from model import build_model, get_normalization_layer, train_model

class HPModelBuilder:

	def __init__(self, normalizer):
		self.normalizer = normalizer
	
	def build_model_from_hyperparameters (self, 
		hyperparameters: kt.engine.hyperparameters.HyperParameters) -> tf.keras.Sequential:

		# define solution space for fixed-length hyperparameters
		hidden_layers = hyperparameters.get("hidden_layers")
		learning_rate = hyperparameters.get("learning_rate")

		# define solution space for variable-length hyperparameters
		units = []

		for i in range(hidden_layers):
			units.append(hyperparameters.get(f"units_{i}"))
		
		return build_model(self.normalizer, hidden_layers, units, learning_rate)

def load_data(data_file: str) -> pd.DataFrame:
	return pd.read_csv(data_file, index_col="Datetime", parse_dates=True)

def generate_search_space(min_hidden_layers: int,
	max_hidden_layers: int,
	min_learning_rate: int,
	max_learning_rate: int,
	hidden_layer_size_options: int) -> kt.HyperParameters:
	
	hyperparameters = kt.HyperParameters()
		
	# define search space for fixed-length hyperparameters
	hidden_layers = hyperparameters.Int("hidden_layers", 
		min_hidden_layers, 
		max_hidden_layers)
	learning_rate = hyperparameters.Float("learning_rate", 
		min_value=min_learning_rate, 
		max_value=max_learning_rate, 
		sampling="log")
		
	for i in range(max_hidden_layers):
		hyperparameters.Choice(f"units_{i}", hidden_layer_size_options)
	
	return hyperparameters

def get_tuner(model_builder: HPModelBuilder, 
	hp_search_space: kt.HyperParameters,
	search_dir: str, 
	search_name: str, 
	search_trials:int) -> kt.BayesianOptimization:
	'''
	Create or load an instance of a keras hyperparameter tuning object.
	For a given tuner, no overwrite will occur (even with changed parameters).
	''' 

	# define search parameters
	tuner = kt.BayesianOptimization(
		model_builder.build_model_from_hyperparameters,
		objective='val_loss',
		max_trials=search_trials,
		directory=search_dir,
		project_name=search_name,
		hyperparameters=hp_search_space)

	return tuner

def search(tuner: kt.BayesianOptimization, 
	train_features: pd.DataFrame, 
	train_labels: pd.DataFrame, 
	val_features: pd.DataFrame, 
	val_labels: pd.DataFrame,
	max_epochs: int,
	early_stopping_patience: int):

	# callback
	early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(tuner.directory, tuner.project_name))

	# search
	tuner.search(train_features, 
		train_labels,
		epochs=max_epochs,
		validation_data=(val_features, val_labels),
		verbose=False,
		callbacks=[early_stopping_callback, tensorboard_callback])

	return None

def get_best_hyperparameters(tuner: kt.BayesianOptimization) -> kt.HyperParameters:
	return tuner.get_best_hyperparameters()[0]

def get_best_model(tuner: kt.BayesianOptimization) -> tf.keras.Sequential:
	return tuner.get_best_models()[0]

def get_best_trial_id(tuner: kt.BayesianOptimization) -> int:
	return tuner.oracle.get_best_trials()[0].trial_id

def extract_hyperparameters_to_series(hyperparameters: kt.HyperParameters) -> pd.Series:

	hp_series = pd.Series(dtype=object)
	hp_series["hidden_layers"] = int(hyperparameters.get("hidden_layers"))
	hp_series["learning_rate"] = f"{hyperparameters.get('learning_rate'):.2E}"

	for i in range(int(hp_series["hidden_layers"])):
		hp_series[f"units_layer_{i}"] = int(hyperparameters.get(f"units_{i}"))

	return hp_series

def extract_history_to_dataframe(history: tf.keras.callbacks.History) -> pd.DataFrame:
	return pd.DataFrame.from_dict(history.history)

def hyperparameter_search(config: dict) -> None:

	# load data
	train_features = load_data(config["train_features_file"])
	train_labels = load_data(config["train_labels_file"])
	val_features = load_data(config["val_features_file"])
	val_labels = load_data(config["val_labels_file"])

	# get hyperparameters
	hp_search_space = generate_search_space(config["hp_min_hidden_layers"],
		config["hp_max_hidden_layers"],
		config["hp_min_learning_rate"],
		config["hp_max_learning_rate"],
		config["hp_hidden_layer_size_choices"])

	# get normalizer 
	normalizer = get_normalization_layer(train_features)
	model_builder = HPModelBuilder(normalizer)

	# get tuner
	tuner = get_tuner(model_builder,
		hp_search_space,
		config["hyperparameter_search_dir"],
		config["hyperparameter_search_name"],
		config["hp_search_trials"])

	# search
	search(tuner, 
		train_features, 
		train_labels,
		val_features,
		val_labels,
		config["ann_max_epochs"],
		config["ann_early_stopping_patience"])

	# get metadata from best model
	best_hyperparameters = get_best_hyperparameters(tuner)
	best_hyperparameters_series = extract_hyperparameters_to_series(best_hyperparameters)
	best_hyperparameters_series.to_csv(config["ann_summary_file"], header=False)
	
	# save model, history, and hyperparameters
	if not os.path.exists(config["ann_model_file"]):
		model = model_builder.build_model_from_hyperparameters(best_hyperparameters)
		history = train_model(model, 
			train_features, 
			train_labels, 
			val_features, 
			val_labels,
			config["ann_max_epochs"],
			config["ann_early_stopping_patience"])

		model.save(config["ann_model_file"])
		history_df = extract_history_to_dataframe(history)
		history_df.to_csv(config["ann_history_file"])	

	return None
