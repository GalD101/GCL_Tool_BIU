#!/bin/bash
# Start Jupyter Notebook without a password or token
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &
