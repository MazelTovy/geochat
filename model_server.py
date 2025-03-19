#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import uvicorn
import os
import logging
import time
from typing import List, Dict, Any, Optional

# Import necessary libraries for LLaMA-3-8B
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_server")

# Create FastAPI application
app = FastAPI(title="Large Language Model API", description="Web API for large language model chat service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, this should be set to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    use_retrieval: Optional[bool] = False
    query_for_retrieval: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    status: str
    retrieval_sources: Optional[List[Dict[str, Any]]] = None
    
# Constants
DEFAULT_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # Long live DeepSeek!
MAX_GPU_MEMORY = "13GiB"  # Adjust based on your GPU capacity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
def load_model():
    logger.info(f"Loading DeepSeek-R1 model on {DEVICE}...")
    
    start_time = time.time()
    
    try:
        # Set quantization if needed to fit in GPU memory
        if DEVICE == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_MODEL_PATH,
                torch_dtype=torch.float16,  # Use half precision for efficiency
                load_in_4bit=True,  # For 4-bit quantization
                device_map="auto",
                trust_remote_code=True,
                use_auth_token=None,
                token=None,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_MODEL_PATH,
                device_map="auto",
                trust_remote_code=True,
                use_auth_token=None,
                token=None,
            )
            
        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_MODEL_PATH,
            use_auth_token=None,
            token=None,
        )
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds!")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

# Format messages for LLaMA-3 input
def format_messages(messages):
    # DeepSeek-R1 can use the built-in chat template
    # This is simpler than manually formatting
    return messages  # Return messages directly, tokenizer will handle formatting

# Generate response
def generate_response(messages, model, tokenizer, max_length=2048, temperature=0.7, top_p=0.9):
    try:
        # Format using built-in chat template
        chat_template = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = tokenizer(chat_template, return_tensors="pt").to(DEVICE)
        
        # Set generation parameters
        generation_kwargs = {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        
        # Generate response
        logger.info("Generating response...")
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                **generation_kwargs
            )
            
        # Decode output
        output = tokenizer.decode(generation_output[0], skip_special_tokens=False)
        
        # Add logging for raw output
        logger.info(f"Raw output: {output}")
        
        # Extract only the assistant response
        # Find the last assistant segment
        response_start = output.rfind("<|assistant|>") + len("<|assistant|>\n")
        response_end = output.rfind("</s>", response_start)
        
        # Add logging for response extraction
        logger.info(f"Response extraction: start={response_start}, end={response_end}")
        
        if response_end == -1:
            response = output[response_start:]
        else:
            response = output[response_start:response_end]
        
        logger.info(f"Final extracted response: {response.strip()}")
        return response.strip()
    except Exception as e:
        logger.error(f"Error occurred while generating response: {str(e)}")
        logger.exception("Detailed traceback:")  # 添加完整堆栈跟踪
        raise e

# Application startup event
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    # Note: In production, you might want to load the model in a separate process and use some form of IPC
    model, tokenizer = load_model()

# API endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Handle retrieval if requested
        if request.use_retrieval and request.query_for_retrieval:
            # Retrieve relevant documents
            retrieval_results = retriever.retrieve(request.query_for_retrieval)
            
            # Create context from retrieved documents
            context = "Retrieved information:\n"
            for result in retrieval_results:
                context += f"- {result['content']}\n"
            
            # Add system message with context
            augmented_messages = [
                {"role": "system", "content": f"Use the following information to answer the user's question: {context}"}
            ] + request.messages
            
            # Generate response with context-augmented messages
            response = generate_response(
                augmented_messages,
                model,
                tokenizer,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            return ChatResponse(
                response=response,
                status="success",
                retrieval_sources=retrieval_results
            )
        else:
            # Regular model response without retrieval
            response = generate_response(
                request.messages, 
                model, 
                tokenizer, 
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            return ChatResponse(
                response=response, 
                status="success"
            )
    except Exception as e:
        logger.error(f"Error in /api/chat: {str(e)}")
        logger.exception("Detailed traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "device": DEVICE}

# Main function
if __name__ == "__main__":
    # Get port from environment variable, default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=port)

# 在文件顶部添加
import os
os.environ["HF_HOME"] = "/scratch/sx2490/huggingface_cache"  # 使用scratch目录，通常有更大配额

# 确保该目录存在
os.makedirs("/scratch/sx2490/huggingface_cache", exist_ok=True) 