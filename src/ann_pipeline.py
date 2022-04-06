import argparse
import json
import os

from download_data import download_data
from clean_data import clean_data
from process_data import process_data
from hyperparameter_search import hyperparameter_search
from evaluate import evaluate

def main(config: dict) -> None:

	outdated = False

	# download
	if not os.path.exists(config["raw_demand_file"]):
		download_data(config["download_demand_url"], config["raw_demand_file"])
		outdated = True

	if not os.path.exists(config["raw_temp_file"]):
		download_data(config["download_temp_url"], config["raw_temp_file"])
		outdated = True

	# clean
	if not os.path.exists(config["cleaned_data_file"]) or outdated:
		clean_data(config["raw_demand_file"], 
					config["raw_temp_file"], 
					config["cleaned_data_file"],
					config["years"])

	# process
	processed_files = [config[f"{a}_{b}_file"] for a in ["train", "val", "test"] for b in ["features", "labels"]]

	if any([not os.path.exists(p) for p in processed_files]) or outdated:
		process_data(config["cleaned_data_file"],
					config["train_features_file"],
					config["train_labels_file"],
					config["val_features_file"],
					config["val_labels_file"],
					config["test_features_file"],
					config["test_labels_file"])

	# hyperparamter search
	hyperparameter_search(config)
	
	# evaluate
	evaluate(config)
	
	return None

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("config_file", type=str, help="File with project configurations (see default_config.json)")

	# parse configs
	args = parser.parse_args()

	with open(args.config_file,'r') as config_input:
		config = json.load(config_input)

	# type conversions
	config["years"] = [int(year) for year in config["years"]]
	config["hp_min_learning_rate"] = float(config["hp_min_learning_rate"])
	config["hp_max_learning_rate"] = float(config["hp_max_learning_rate"])
	config["hp_min_hidden_layers"] = int(config["hp_min_hidden_layers"])
	config["hp_max_hidden_layers"] = int(config["hp_max_hidden_layers"])
	config["hp_hidden_layer_size_choices"] = [int(size) for size in config["hp_hidden_layer_size_choices"]]
	config["hp_search_trials"] = int(config["hp_search_trials"])
	config["ann_max_epochs"] = int(config["ann_max_epochs"])
	config["ann_early_stopping_patience"] = int(config["ann_early_stopping_patience"])
	
	main(config)

	