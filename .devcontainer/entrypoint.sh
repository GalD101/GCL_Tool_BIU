#!/bin/bash
# Start Jupyter Notebook with a password
jupyter notebook password <<EOF
$(cat .devcontainer/password.txt)
EOF

# Start Jupyter Notebook in the background
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
