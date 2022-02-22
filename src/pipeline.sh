# !/bin/bash

# input check
if [ "$#" -ne 1 ]; then
    echo "Missing arguments for pipeline.sh."
	echo ""
	echo "pipeline.sh <bal_authority>"
	exit 1
fi

# input parameters
BAL_AUTHORITY=$1
YEAR_FROM=2016
YEAR_TO=2019

# file structure
THIS_FILE_PATH=$(realpath $0)
THIS_DIR=$(dirname $THIS_FILE_PATH)
DATA_DIR=$THIS_DIR/../data
RAW_DEMAND_FILE_PATH=$DATA_DIR/00-raw/$BAL_AUTHORITY-demand.csv
RAW_TEMPERATURE_FILE_PATH=$DATA_DIR/00-raw/$BAL_AUTHORITY-temperature.csv
CLEANED_DATA_FILE_PATH=$DATA_DIR/01-cleaned/$BAL_AUTHORITY-cleaned.csv
PROCESSED_DATA_FILE_PATH=$DATA_DIR/02-processed/$BAL_AUTHORITY-processed.csv
HYPERPARAMETER_SEARCH_DIR=$DATA_DIR/03-models/hyperparameter-search/
MODEL_FILE_PATH=$DATA_DIR/03-models/$BAL_AUTHORITY.model
RESULTS_FILE_PATH=$DATA_DIR/04-results/model-results.csv

set -e

# download demand 
if [ ! -f $RAW_DEMAND_FILE_PATH ]; then
	python download_demand.py $BAL_AUTHORITY $RAW_DEMAND_FILE_PATH
fi

# clean
if [ ! -f $CLEANED_DATA_FILE_PATH ]; then
	python clean_data.py $RAW_DEMAND_FILE_PATH $RAW_TEMPERATURE_FILE_PATH $CLEANED_DATA_FILE_PATH $YEAR_FROM $YEAR_TO
fi

# process
if [ ! -f $PROCESSED_DATA_FILE_PATH ]; then
	python process_data.py $CLEANED_DATA_FILE_PATH $PROCESSED_DATA_FILE_PATH
fi

python hyperparameter_search.py $PROCESSED_DATA_FILE_PATH $HYPERPARAMETER_SEARCH_DIR $BAL_AUTHORITY $MODEL_FILE_PATH
python evaluate.py $PROCESSED_DATA_FILE_PATH $HYPERPARAMETER_SEARCH_DIR $BAL_AUTHORITY $RESULTS_FILE_PATH
