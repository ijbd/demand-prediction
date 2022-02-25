import pandas as pd
import numpy as np
import tensorflow as tf 

def build_model(normalizer,hidden_layers,units,dropout,learning_rate):
	'''
	Build and compile tensorflow multi-layer perceptron.

	:param normalizer: Tensorflow normalization layer.
	:type normalizer: keras.layers.preprocessing.normalization.Normalization
	:param hidden_layers: Number of hidden layers to include
	:type hidden_layers: int
	:param units: List defining the number of neurons for each successive layer.
	:type units: list (int)
	:param dropout: List defining the training dropout rate for each successive layer.
	:type dropout: list (float)
	:returns: Tensorflow model
	:rtype: keras.engine.sequential.Sequential
	'''

	# initialize model
	model = tf.keras.Sequential()

	# add input layer (with normalization)
	model.add(normalizer)

	# add hidden layers	
	for i in range(hidden_layers):
		model.add(tf.keras.layers.Dense(units=units[i],activation='relu'))
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
	Split data into train, validation, and test dataset. 
	To include data spanning the whole time range in each 
	dataset while ensuring correct seasonal distributions,
	months are shuffled between years creating a 50/25/25 split.

	:param processed_data: Processed data with datetime index.
	:type processed_data: pd.DataFrame
	:returns: Train, validation, and test datasets.
	:rtype: tuple (pd.DataFrame)
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
		test_dataset = pd.concat((test_dataset,processed_data[(processed_data.index.year.isin(years[month][:num_training_years])) & (processed_data.index.month == month+1)]))
		validation_dataset = pd.concat((validation_dataset, processed_data[(processed_data.index.year.isin(years[month][num_training_years:])) & (processed_data.index.month == month+1)]))
		
	train_dataset = processed_data[~processed_data.index.isin(test_dataset.index.append(validation_dataset.index))]

	return train_dataset, validation_dataset, test_dataset

def split_features(dataset):
	'''
	Return features and labels.

	:param dataset: Dataset with demand labels in column 'D'.
	:type dataset: pd.DataFrame
	:returns: Features and labels as separate DataFrames.
	:rtype: tuple (pd.DataFrame)
	'''
	features = dataset.copy()
	labels = features.pop('D')

	return features, labels

def get_normalizer(train_features):
	'''
	Return a tensor flow normalization layer for a set of features

	:param train_features: Features dataset to fit.
	:type train_features: pd.DataFrame or np.ndarrary
	:returns: Normalization layer
	:rtype: keras.layers.preprocessing.normalization.Normalization
	'''
	normalizer = tf.keras.layers.Normalization(axis=-1)
	normalizer.adapt(np.array(train_features))

	return normalizer

def train_model(model, train_features, train_labels, val_features, val_labels):
	'''
	Trains model for 5000 epochs (with early stopping decided using validation data).

	:param model: Model to train.
	:type model: keras.engine.sequential.Sequential
	:param train_features: Training features compatible with model architecture.
	:type train_features: pd.DataFrame
	:param train_labels: Training labels compatible with model architecture.
	:type train_labels: pd.DataFrame
	:param val_features: Validation features compatible with model architecture.
	:type val_features: pd.DataFrame
	:param val_labels: Validation labels compatible with model architecture.
	:type val_labels: pd.DataFrame
	'''
	
	# add early stopping 
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

	# train
	history = model.fit(train_features,
						train_labels,
						validation_data=(val_features, val_labels),
						epochs=5000, 
						batch_size=1, 
						callbacks=[early_stop],
						verbose=0)

	return history
