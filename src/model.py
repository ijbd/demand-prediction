import pandas as pd
import numpy as np
import tensorflow as tf 

def build_model(normalizer: tf.keras.layers.Normalization, hidden_layers: int, units: list, learning_rate: float) -> tf.keras.Sequential:
	
	# initialize model
	model = tf.keras.Sequential()

	# add input layer (with normalization)
	model.add(normalizer)

	# add hidden layers	
	for i in range(hidden_layers):
		model.add(tf.keras.layers.Dense(units=units[i], activation='relu'))
	
	# add output layer
	model.add(tf.keras.layers.Dense(1))

	# compile model
	model.compile(loss='mean_absolute_error', optimizer = tf.keras.optimizers.Adam(learning_rate))

	return model

def get_normalization_layer(features: pd.DataFrame) -> tf.keras.layers.Normalization:

	normalizer = tf.keras.layers.Normalization(axis=-1)
	normalizer.adapt(np.array(features))

	return normalizer

def train_model(model: tf.keras.Sequential, 
				train_features: pd.DataFrame, 
				train_labels: pd.DataFrame, 
				val_features: pd.DataFrame, 
				val_labels: pd.DataFrame, 
				max_epochs: int,
				early_stopping_patience: int): #TODO
	
	# add early stopping 
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

	# train
	history = model.fit(train_features,
						train_labels,
						validation_data=(val_features, val_labels),
						epochs=max_epochs, 
						batch_size=1, 
						callbacks=[early_stop],
						verbose=1)

	return history
