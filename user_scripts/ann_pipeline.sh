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

# file structure
REPO_DIR=$(dirname $(dirname $(realpath $0)))
PROJECT_DIR=/nfs/turbo/seas-mtcraig/ijbd/demand_ml
RAW_DATA_DIR=$PROJECT_DIR/data/00_raw
CLEANED_DATA_DIR=$PROJECT_DIR/data/01_cleaned
PROCESSED_DATA_DIR=$PROJECT_DIR/data/02_processed
MODEL_DIR=$PROJECT_DIR/data/03_models/$BAL_AUTH
CONFIG_FILE=$MODEL_DIR/config.json

# setup file structure
if [ ! -d $PROJECT_DIR ]; then
    echo "Project directory does not exist: $PROJECT_DIR"
    exit 1
fi

mkdir -p $RAW_DATA_DIR
mkdir -p $CLEANED_DATA_DIR
mkdir -p $PROCESSED_DATA_DIR
mkdir -p $MODEL_DIR

# generate config file
DEFAULT_CONFIG_FILE=$REPO_DIR/config/default_config.json
sed "s#{PROJECT_DIR}#$PROJECT_DIR#g" $DEFAULT_CONFIG_FILE | sed -e "s#{BAL_AUTH}#$BAL_AUTH#g" > $CONFIG_FILE

# python script
python $REPO_DIR/src/ann_pipeline.py $CONFIG_FILE
