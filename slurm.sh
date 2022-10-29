#!/bin/bash

# run the script
cd $SLURM_SUBMIT_DIR # should be .../cryptic-crossword-rationale/

# activate python virtual environment
source env/bin/activate

# run the script
# if [ $# != 2 ]
# then
#     echo "pass experiment_folder and configuration_file as arguments"
#     exit
# fi
python3 src/main.py

# deactivate python virtual environment
deactivate
