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
#SBATCH --job-name=deepseek_server

# Create log directory
mkdir -p /scratch/sx2490/Logs/model_server

# Clean modules
module purge

# Set server port (change to an unused port if needed)
export PORT=8000
export STATIC_PORT=8080  # Static server port

# Create node and port information file for reference
NODEFILE="/scratch/sx2490/Logs/model_server/node_info.txt"
echo "Job ID: $SLURM_JOB_ID" > $NODEFILE
echo "Running on node: $SLURMD_NODENAME" >> $NODEFILE
echo "API service port: $PORT" >> $NODEFILE
echo "Static server port: $STATIC_PORT" >> $NODEFILE
echo "Start time: $(date)" >> $NODEFILE

# Print node information (for SSH port forwarding)
echo "================ SERVER INFORMATION ================"
echo "Running on node: $SLURMD_NODENAME"
echo "API server port: $PORT"
echo "Static server port: $STATIC_PORT"
echo "Use the following command for SSH port forwarding:"
echo "ssh -L ${PORT}:${SLURMD_NODENAME}:${PORT} -L ${STATIC_PORT}:${SLURMD_NODENAME}:${STATIC_PORT} sx2490@greene.hpc.nyu.edu"
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
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128; \
    \
    # Install base dependencies
    pip install --user fastapi uvicorn pydantic -q; \
    pip install --user transformers accelerate bitsandbytes sentencepiece -q; \
    \
    # Install FlashRAG dependencies - including those missing from previous run
    pip install --user huggingface_hub sentence-transformers -q; \
    pip install --user datasets einops tiktoken -q; \
    pip install --user faiss-cpu faiss-gpu scipy numpy tqdm -q; \
    pip install --user langchain openai -q; \
    pip install --user langid bm25s -q; \
    \
    # If FlashRAG is not installed, try to install it here
    if [ ! -d /scratch/sx2490/FlashRAG ]; then \
        echo 'FlashRAG not found, cloning repository...'; \
        cd /scratch/sx2490 && \
        git clone https://github.com/RUC-NLPIR/FlashRAG.git && \
        cd FlashRAG && \
        pip install --user -e . -q; \
        echo 'FlashRAG installed successfully.'; \
    else \
        echo 'FlashRAG already installed at /scratch/sx2490/FlashRAG'; \
    fi; \
    \
    # Add FlashRAG to Python path
    export PYTHONPATH=/scratch/sx2490:/scratch/sx2490/FlashRAG:\$PYTHONPATH; \
    cd /scratch/sx2490 && \
    \
    # Check if data needs to be converted to a standard corpus format for FlashRAG
    echo 'Checking if corpus conversion is needed...' && \
    CORPUS_DIR=/scratch/sx2490/geochat_gs/nyc_schools_data && \
    PROCESSED_DIR=\$CORPUS_DIR/processed && \
    CORPUS_JSONL=\$PROCESSED_DIR/corpus.jsonl && \
    mkdir -p \$PROCESSED_DIR && \
    \
    # Use separate Python script for corpus conversion
    if [ ! -f \$CORPUS_JSONL ]; then \
        echo 'Converting NYC schools data to corpus format...' && \
        cd /scratch/sx2490/geochat_gs && \
        python convert_corpus.py --corpus-dir \$CORPUS_DIR --output-file \$CORPUS_JSONL && \
        echo 'Conversion complete'; \
    else \
        echo 'Corpus file already exists: '\$CORPUS_JSONL; \
        echo 'Corpus file content check:'; \
        head -n 2 \$CORPUS_JSONL; \
        echo '...'; \
        wc -l \$CORPUS_JSONL; \
    fi && \
    \
    # Verify corpus file exists and has valid content
    if [ ! -f \$CORPUS_JSONL ]; then \
        echo 'ERROR: Corpus file not created properly!'; \
        exit 1; \
    fi && \
    \
    # Check if RAG index exists and build it if needed
    echo 'Checking if RAG initialization is needed...' && \
    INDEX_PATH=\$PROCESSED_DIR/index && \
    if [ ! -d \$INDEX_PATH ]; then \
        echo 'RAG index not found. Building index...' && \
        cd /scratch/sx2490 && \
        # Use direct command, printing the exact command
        echo 'Running: python -m flashrag.retriever.index_builder --retrieval_method e5 --model_path intfloat/e5-base-v2 --corpus_path '\$CORPUS_JSONL' --save_dir '\$PROCESSED_DIR' --use_fp16 --max_length 512 --batch_size 32 --sentence_transformer --faiss_type Flat' && \
        \
        python -m flashrag.retriever.index_builder \
            --retrieval_method e5 \
            --model_path intfloat/e5-base-v2 \
            --corpus_path \$CORPUS_JSONL \
            --save_dir \$PROCESSED_DIR \
            --use_fp16 \
            --max_length 512 \
            --batch_size 32 \
            --sentence_transformer \
            --faiss_type Flat && \
        echo 'RAG index creation completed.'; \
        \
        # Verify index was created
        if [ ! -d \$INDEX_PATH ]; then \
            echo 'ERROR: Index creation failed. Index directory not found!'; \
            echo 'Listing contents of processed directory:'; \
            ls -la \$PROCESSED_DIR; \
            echo 'Falling back to basic retrieval without FlashRAG.'; \
        else \
            echo 'Index successfully created at: '\$INDEX_PATH; \
            ls -la \$INDEX_PATH; \
        fi; \
    else \
        echo 'RAG index already exists at: '\$INDEX_PATH; \
        ls -la \$INDEX_PATH; \
    fi && \
    \
    # Start both servers simultaneously
    echo 'Starting both API and static servers...' && \
    # Start static server in background
    cd /scratch/sx2490/geochat_gs && \
    echo 'Starting static server on port $STATIC_PORT...' && \
    python static_server.py --port $STATIC_PORT > /scratch/sx2490/Logs/model_server/static_server.log 2>&1 & \
    STATIC_PID=\$! && \
    echo 'Static server started with PID: '\$STATIC_PID && \
    \
    # Start the model server
    echo 'Starting model server on port $PORT...' && \
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