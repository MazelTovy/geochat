#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --account=pr_325_general
#SBATCH --output=/scratch/sx2490/Logs/model_server/%j.out
#SBATCH --error=/scratch/sx2490/Logs/model_server/%j.err
#SBATCH --job-name=llama3_server

# Create log directory
mkdir -p /scratch/sx2490/Logs/model_server

# Clean modules
module purge

# Set server port (change to an unused port if needed)
export PORT=8000

# Create node and port information file for reference
NODEFILE="/scratch/sx2490/Logs/model_server/node_info.txt"
echo "Job ID: $SLURM_JOB_ID" > $NODEFILE
echo "Running on node: $SLURMD_NODENAME" >> $NODEFILE
echo "Service port: $PORT" >> $NODEFILE
echo "Start time: $(date)" >> $NODEFILE

# Print node information (for SSH port forwarding)
echo "================ SERVER INFORMATION ================"
echo "Running on node: $SLURMD_NODENAME"
echo "Using port: $PORT"
echo "Use the following command for SSH port forwarding:"
echo "ssh -L ${PORT}:${SLURMD_NODENAME}:${PORT} sx2490@greene.hpc.nyu.edu"
echo "===========================================" 

# Create Python package and cache directories in scratch
SCRATCH_PYTHON_DIR="/scratch/sx2490/python_packages"
SCRATCH_HF_CACHE="/scratch/sx2490/huggingface_cache"
SCRATCH_TORCH_CACHE="/scratch/sx2490/torch_cache"
SCRATCH_TRANSFORMERS_CACHE="/scratch/sx2490/transformers_cache"
SCRATCH_PIP_CACHE="/scratch/sx2490/pip_cache"

mkdir -p $SCRATCH_PYTHON_DIR
mkdir -p $SCRATCH_HF_CACHE
mkdir -p $SCRATCH_TORCH_CACHE
mkdir -p $SCRATCH_TRANSFORMERS_CACHE
mkdir -p $SCRATCH_PIP_CACHE

# Clear existing home directory caches
echo "Cleaning home directory caches..."
rm -rf ~/.local/lib/python*
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch
rm -rf ~/.cache/transformers

# Set all relevant environment variables to use scratch directories
export PYTHONUSERBASE=$SCRATCH_PYTHON_DIR
export PATH="$SCRATCH_PYTHON_DIR/bin:$PATH"
export HF_HOME=$SCRATCH_HF_CACHE
export TORCH_HOME=$SCRATCH_TORCH_CACHE
export TRANSFORMERS_CACHE=$SCRATCH_TRANSFORMERS_CACHE
export PIP_CACHE_DIR=$SCRATCH_PIP_CACHE

# Add these lines to ensure environment variables are cleared
unset HUGGING_FACE_HUB_TOKEN
unset HUGGINGFACE_HUB_TOKEN
unset HF_TOKEN

# Configure FlashRAG path
export PYTHONPATH="/scratch/sx2490:/scratch/sx2490/FlashRAG:$PYTHONPATH"

# Also set environment variables in singularity command
singularity exec --nv \
    --overlay /scratch/sx2490/pytorch-example/my_pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    # Set environment variables inside singularity container
    export PYTHONUSERBASE=$SCRATCH_PYTHON_DIR; \
    export PATH=$SCRATCH_PYTHON_DIR/bin:\$PATH; \
    export HF_HOME=$SCRATCH_HF_CACHE; \
    export TORCH_HOME=$SCRATCH_TORCH_CACHE; \
    export TRANSFORMERS_CACHE=$SCRATCH_TRANSFORMERS_CACHE; \
    export PIP_CACHE_DIR=$SCRATCH_PIP_CACHE; \
    \
    # Install all required dependencies
    pip install --user fastapi uvicorn pydantic -q; \
    pip install --user transformers accelerate bitsandbytes sentencepiece -q; \
    pip install --user huggingface_hub sentence-transformers -q; \
    pip install --user datasets einops tiktoken -q; \
    pip install --user faiss-cpu scipy numpy tqdm -q; \
    pip install --user langchain -q; \
    \
    # If FlashRAG is not installed, try to install it here
    if [ ! -d /scratch/sx2490/FlashRAG ]; then \
        cd /scratch/sx2490 && \
        git clone https://github.com/RUC-NLPIR/FlashRAG.git && \
        cd FlashRAG && \
        pip install --user -e . -q; \
    fi; \
    \
    export PYTHONPATH=/scratch/sx2490:/scratch/sx2490/FlashRAG:\$PYTHONPATH; \
    cd /scratch/sx2490 && \
    \
    # Check if RAG initialization is needed and run if necessary
    echo 'Checking if RAG initialization is needed...' && \
    PROCESSED_DIR=/scratch/sx2490/geochat_gs/nyc_schools_data/processed && \
    INDEX_PATH=\$PROCESSED_DIR/index && \
    CORPUS_PATH=\$PROCESSED_DIR/corpus.jsonl && \
    if [ ! -d \$INDEX_PATH ] || [ ! -f \$CORPUS_PATH ]; then \
        echo 'RAG index or corpus not found. Running RAG initialization...' && \
        cd /scratch/sx2490/geochat_gs && \
        python init_rag.py --docs-dir nyc_schools_data --chunk-size 500 --chunk-overlap 50 --embedding-model intfloat/e5-base-v2; \
        echo 'RAG initialization completed.'; \
    else \
        echo 'RAG index and corpus already exist. Skipping initialization.'; \
    fi && \
    \
    # Start the model server
    echo 'Starting model server...' && \
    python -m geochat_gs.model_server"

# Record end time
echo "End time: $(date)" >> $NODEFILE

# Check job status
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Reason

# Clean up home directory caches (run this at the end to clear any remaining caches)
echo "Final cleanup of home directory caches..."
rm -rf ~/.local/lib/python*
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch
rm -rf ~/.cache/transformers