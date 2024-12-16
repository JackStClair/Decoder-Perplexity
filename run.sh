#!/bin/bash

{
    unzip hw3.zip -d new_dir  # unzip into a new directory

    cd new_dir  # go to the new directory

    python -m venv venv  # create a virtual environment

    source venv/bin/activate  # activate the virtual environment

    pip install -r requirements.txt  # install dependencies


    echo '========== start running =========='  # start running the main file
    python run.py submission.csv

} 2>&1 | tee record.txt  # record all outputs