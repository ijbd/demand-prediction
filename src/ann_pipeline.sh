# !/bin/bash

# input check
if [ "$#" -ne 1 ]; then
    echo "Missing arguments for setup_pipeline.sh"
	echo ""
	echo "bash setup_pipeline.sh <bal_authority>"
	exit 1
fi

# parameters
BAL_AUTH=$1
DEFAULT_CONFIG_FILE=default_config.json

# file structure
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
RAW_DATA_DIR=$PROJECT_DIR/data/00_raw
CLEANED_DATA_DIR=$PROJECT_DIR/data/01_cleaned
PROCESSED_DATA_DIR=$PROJECT_DIR/data/02_processed
MODEL_DIR=$PROJECT_DIR/data/03_models/$BAL_AUTH
CONFIG_FILE=$MODEL_DIR/config.json

# setup file structure
mkdir -p $RAW_DATA_DIR
mkdir -p $CLEANED_DATA_DIR
mkdir -p $PROCESSED_DATA_DIR
mkdir -p $MODEL_DIR

# generate config file
sed "s#{PROJECT_DIR}#$PROJECT_DIR#g" $DEFAULT_CONFIG_FILE | sed -e "s#{BAL_AUTH}#$BAL_AUTH#g" > $CONFIG_FILE

# python script
python ann_pipeline.py $CONFIG_FILE