#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --account=pr_325_general
#SBATCH --output=/home/sx2490/Logs/model_server/%j.out
#SBATCH --error=/home/sx2490/Logs/model_server/%j.err
#SBATCH --job-name=llama3_server

# Create log directory
mkdir -p /home/sx2490/Logs/model_server

# Clean modules
module purge

# Set server port (change to an unused port if needed)
export PORT=8000

# Create node and port information file for reference
NODEFILE="/home/sx2490/Logs/model_server/node_info.txt"
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

# Run the service using singularity container

# 添加这些行，确保环境变量被清除
unset HUGGING_FACE_HUB_TOKEN
unset HUGGINGFACE_HUB_TOKEN
unset HF_TOKEN

# 在singularity命令之前添加
export HF_HOME="/scratch/sx2490/huggingface_cache"
mkdir -p $HF_HOME

# 在singularity命令中也设置
singularity exec --nv \
    --overlay /scratch/sx2490/pytorch-example/my_pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    pip install fastapi uvicorn pydantic -q; \
    pip install transformers accelerate bitsandbytes sentencepiece -q; \
    pip install huggingface_hub -q; \
    export PATH=/home/sx2490/.local/bin:\$PATH; \
    export HF_HOME=/scratch/sx2490/huggingface_cache; \
    cd /home/sx2490 && \
    python -m geochat_gs.model_server"

# Record end time
echo "End time: $(date)" >> $NODEFILE

# Check job status
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Reason