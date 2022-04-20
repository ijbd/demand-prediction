import shutil
import os
import pandas
import argparse

def compile_results(project_dir, compilation_dir):
    all_models_dir = os.path.join(project_dir, "data", "03_models")
    
    for model_name in os.listdir(all_models_dir):
        model_dir = os.path.join(all_models_dir, model_name)

        # copy history file and model
        source_path = os.path.join(model_dir, f"ann_history.csv")
        dest_path = os.path.join(compilation_dir, f"{model_name}_ann_history.csv")
        shutil.copy(source_path, dest_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Top-level project directory (see 'default_config.json').")
    parser.add_argument("compilation_dir", type=str, help="Directory to compile all results.")

    args = parser.parse_args()
    compile_results(args.project_dir, args.compilation_dir)
