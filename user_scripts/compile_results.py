import os
import pandas
import argparse

def main(project_dir):

	all_models_dir = os.path.join(project_dir, "data", "03_models")

	for model_name in os.listdir(all_models_dir):
		model_dir = os.path.join(all_models_dir, model_name)

		# compile summary, history, and hyperparameter files
		summary_file = 

	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("project_dir", type=str, help="Top-level project directory (see 'default_config.json').")
	parser.add_argument("compilation_dir", type=str, help="Directory to compile all results.")

	args = parser.parse_args()

	main(args.project_dir)
