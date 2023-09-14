#!/bin/bash

# Path to the script to be executed
script_to_execute="./your_script.sh"

# Initialize the first argument at 0
arg1=0

# Second argument is always 500
arg2=500

# Loop from 0 to 29500 with steps of 500
while [ $arg1 -le 29500 ]; do
    # Execute the other script with the arguments
    sbatch $script_to_execute $arg1 $arg2
    # Increment the first argument by 500
    arg1=$((arg1 + 500))
done