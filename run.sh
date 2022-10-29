#!/bin/bash
# file to manage running of slurm scripts

# expect 2 inputs
if [ $# != 2 ]
then
    echo "Usage: $0 <partition> <experiment folder>"
    echo "Example: $0 batch test"
    exit
fi

# specify the filepath to use
filepath="./models/unet/$2"
# make the directories as needed
mkdir -p "$filepath"

FILE=$filepath/cluster_output.log
if test -f "$FILE"; then
    echo "$FILE exists."
    exit
fi

# run the file
sbatch --partition=$1 --nodes=1 --job-name=$2 --output=$filepath/cluster_output.log --error=$filepath/cluster_error.log slurm.sh
