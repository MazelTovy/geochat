#!/bin/bash

# Quick start script for testing Gemini service on HPC with singularity

echo "=== Gemini Quick Start Script (Singularity Version) ==="

# Set up environment variables
SCRATCH_PYTHON_DIR="/scratch/sx2490/python_packages"
SCRATCH_HF_CACHE="/scratch/sx2490/huggingface_cache"
SCRATCH_TORCH_CACHE="/scratch/sx2490/torch_cache"
SCRATCH_TRANSFORMERS_CACHE="/scratch/sx2490/transformers_cache"
SCRATCH_PIP_CACHE="/scratch/sx2490/pip_cache"

# Create cache directories
mkdir -p $SCRATCH_PYTHON_DIR
mkdir -p $SCRATCH_HF_CACHE
mkdir -p $SCRATCH_TORCH_CACHE
mkdir -p $SCRATCH_TRANSFORMERS_CACHE
mkdir -p $SCRATCH_PIP_CACHE

# Set environment variables
export PYTHONUSERBASE=$SCRATCH_PYTHON_DIR
export PATH="$SCRATCH_PYTHON_DIR/bin:$PATH"
export HF_HOME=$SCRATCH_HF_CACHE
export TORCH_HOME=$SCRATCH_TORCH_CACHE
export TRANSFORMERS_CACHE=$SCRATCH_TRANSFORMERS_CACHE
export PIP_CACHE_DIR=$SCRATCH_PIP_CACHE
export PYTHONPATH="/scratch/sx2490/geochat_gs:$PYTHONPATH"
export GEMINI_API_KEY="Your_api_key"

# Change to geochat_gs directory
cd /scratch/sx2490/geochat_gs

# Check if data directory exists
if [ ! -d "/scratch/sx2490/Chatbot_No_Map/nyc_schools_data" ]; then
    echo "Creating NYC schools data directory..."
    mkdir -p /scratch/sx2490/Chatbot_No_Map/nyc_schools_data
    
    # Copy data if available
    if [ -d "nyc_schools_data" ]; then
        echo "Copying NYC schools data..."
        cp -r nyc_schools_data/* /scratch/sx2490/Chatbot_No_Map/nyc_schools_data/
    fi
fi

# Start Gemini server in singularity environment
echo "Starting Gemini server in singularity environment on port 8001..."

singularity exec --nv \
    --overlay /scratch/sx2490/pytorch-example/my_pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    export PYTHONUSERBASE=$SCRATCH_PYTHON_DIR; \
    export PATH=$SCRATCH_PYTHON_DIR/bin:\$PATH; \
    export HF_HOME=$SCRATCH_HF_CACHE; \
    export HF_TOKEN=\"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\"; \
    export TORCH_HOME=$SCRATCH_TORCH_CACHE; \
    export TRANSFORMERS_CACHE=$SCRATCH_TRANSFORMERS_CACHE; \
    export PIP_CACHE_DIR=$SCRATCH_PIP_CACHE; \
    export PYTHONPATH=\"/scratch/sx2490/geochat_gs:\$PYTHONPATH\"; \
    export GEMINI_API_KEY=\"$GEMINI_API_KEY\"; \
    \
    # Install dependencies if needed
    echo 'Installing Python dependencies...'; \
    pip install --user google-generativeai fastapi uvicorn sentence-transformers scikit-learn pandas numpy; \
    \
    # Start Gemini server
    echo 'Starting Gemini server...'; \
    cd /scratch/sx2490/geochat_gs; \
    python gemini_model_server.py --port 8001 --data_dir /scratch/sx2490/Chatbot_No_Map/nyc_schools_data"

echo "Gemini server stopped." 
