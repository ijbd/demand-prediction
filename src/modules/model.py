import pandas as pd
import numpy as np
import tensorflow as tf 

def build_model(normalizer,hidden_layers,units,activation,dropout,learning_rate):
	'''
	TODO

	:param normalizer: TODO
	:type normalizer: TODO
	:param hidden_layers: TODO
	:type hidden_layers: TODO
	:param units: TODO
	:type units: TODO
	:param activation: TODO
	:type activation: TODO
	:param dropout: TODO
	:type dropout: TODO
	:returns: TODO
	:rtype: TODO
	'''

	# initialize model
	model = tf.keras.Sequential()

	# add input layer (with normalization)
	model.add(normalizer)

	# add hidden layers	
	for i in range(hidden_layers):
		model.add(tf.keras.layers.Dense(units=units[i],activation=activation))
		model.add(tf.keras.layers.Dropout(rate=dropout[i]))
	
	# add output layer
	model.add(tf.keras.layers.Dense(1))

	# compile model
	model.compile(
		loss='mean_absolute_error',
		optimizer = tf.keras.optimizers.Adam(learning_rate)
		)

	return model

def split_data(processed_data):
	'''
	TODO
	'''

	# initialize dataset
	test_dataset = pd.DataFrame()
	validation_dataset = pd.DataFrame()

	# create shuffled months
	np.random.seed(1)

	# find training size
	year_from = np.min(processed_data.index.year)
	year_to = np.max(processed_data.index.year)

	num_training_years = (year_to - year_from + 1) // 4

	years = [np.random.choice(range(year_from,year_to+1),2*num_training_years,replace=False) for i in range(12)]

	for month in range(12):
		test_dataset = test_dataset.append(processed_data[(processed_data.index.year.isin(years[month][:num_training_years])) & (processed_data.index.month == month+1)])
		validation_dataset = validation_dataset.append(processed_data[(processed_data.index.year.isin(years[month][num_training_years:])) & (processed_data.index.month == month+1)])
		
	train_dataset = processed_data[~processed_data.index.isin(test_dataset.index.append(validation_dataset.index))]

	return train_dataset, validation_dataset, test_dataset

def split_features(dataset):
	'''
	Return features and labels.

	:param dataset: TODO
	:type dataset: TODO
	:returns: TODO
	:rtype: TODO
	'''
	features = dataset.copy()
	labels = features.pop('D')

	return features, labels

def get_normalizer(train_features):
	'''
	TODO
	'''
	normalizer = tf.keras.layers.Normalization(axis=-1)
	normalizer.adapt(np.array(train_features))

	return normalizer

def train_model(model, train_features, train_labels, val_features, val_labels):
	# add early stopping 
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

	# train
	model.fit(train_features,
				train_labels,
				validation_data=(val_features, val_labels),
				epochs=10, 
				batch_size=1, 
				callbacks=[early_stop],
				verbose=0)

	return None
