#!/bin/bash

# Gemini Model Server Startup Script

echo "Starting Gemini Model Server..."

# 设置环境变量
export PYTHONPATH=/scratch/sx2490/geochat_gs:$PYTHONPATH

# 请在这里设置您的 Gemini API Key
export GEMINI_API_KEY="AIzaSyCa_W1oSN7oxMSvY4vYj44madQo6JztpRA"

# 检查API Key
if [ "$GEMINI_API_KEY" = "YOUR_GEMINI_API_KEY_HERE" ]; then
    echo "WARNING: Please set your Gemini API key in this script!"
    echo "Edit run_gemini_server.sh and replace YOUR_GEMINI_API_KEY_HERE with your actual API key"
fi

# 安装必要的Python包
echo "Installing required packages..."
pip install google-generativeai fastapi uvicorn sentence-transformers scikit-learn

# 启动服务器
echo "Starting server on port 8001..."
python gemini_model_server.py --port 8001 --data_dir /scratch/sx2490/Chatbot_No_Map/nyc_schools_data 