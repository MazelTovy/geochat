#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import torch
import uvicorn
import os
import logging
import time
import traceback
import re
import json
from typing import List, Dict, Any, Optional
import argparse
import numpy as np

# Import necessary libraries for DeepSeek-R1
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI
import httpx

# Import RAG components
from rag.retriever import DocumentRetriever, format_retrieved_documents

# Set environment variables at the top of the file to use scratch dir
os.environ["HF_HOME"] = "/scratch/sx2490/huggingface_cache"  # Use scratch directory for larger quota
os.makedirs("/scratch/sx2490/huggingface_cache", exist_ok=True)

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key, 
    base_url=openai_api_base,
    http_client=httpx.Client(verify=False)  # 禁用SSL验证
)


# Configure detailed logging
os.makedirs("/scratch/sx2490/Logs/model_server", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/scratch/sx2490/Logs/model_server/api_server.log')
    ]
)
logger = logging.getLogger("model_server")

# Create FastAPI application
app = FastAPI(title="DeepSeek-R1 Chat API", description="Web API for DeepSeek-R1 chat service with RAG capabilities")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Explicitly allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 确保VLLM_SERVER_URL指向正确的端口
VLLM_SERVER_URL = "http://localhost:8000/v1/chat/completions"

async def generate_response_vllm_api(messages, max_length=2048, temperature=0.7, top_p=0.9):
    """Send request to vllm api with reasoning support"""
    payload = {
        "model": "DeepSeek-R1", # 使用服务器启动时的served-model-name
        "messages": messages,
        "max_tokens": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True
    }
    
    try:
        async with httpx.AsyncClient(timeout=300.0, verify=False) as client:
            response = await client.post(VLLM_SERVER_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 从响应中提取内容和reasoning
            content = result["choices"][0]["message"]["content"]
            reasoning_content = result["choices"][0]["message"].get("reasoning_content")
            
            # 可能的logprobs
            logprobs = None
            if "logprobs" in result["choices"][0]:
                logprobs = result["choices"][0]["logprobs"]
            
            logger.info(f"vLLM API response received: {len(content)} chars")
            if reasoning_content:
                logger.info(f"Reasoning content: {len(reasoning_content)} chars")
            
            return content, reasoning_content, logprobs
    except Exception as e:
        logger.error(f"Error in vLLM API call: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

async def generate_response_vllm_api_stream(messages, max_length=2048, temperature=0.7, top_p=0.9):
    """Send request to vllm api with streaming support"""
    payload = {
        "model": "DeepSeek-R1", 
        "messages": messages,
        "max_tokens": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True  # 启用流式输出
    }
    
    full_content = ""
    full_reasoning = ""
    
    try:
        async with httpx.AsyncClient(timeout=300.0, verify=False) as client:
            async with client.stream("POST", VLLM_SERVER_URL, json=payload) as response:
                response.raise_for_status()
                
                async for chunk in response.aiter_lines():
                    if not chunk.strip():
                        continue
                    
                    # 去除 "data: " 前缀
                    if chunk.startswith("data: "):
                        chunk = chunk[6:]
                    
                    # 跳过结束标志
                    if chunk == "[DONE]":
                        break
                    
                    try:
                        chunk_data = json.loads(chunk)
                        delta = chunk_data["choices"][0]["delta"]
                        
                        # 处理内容增量
                        if "content" in delta and delta["content"]:
                            full_content += delta["content"]
                        
                        # 处理推理内容增量
                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            full_reasoning += delta["reasoning_content"]
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode chunk: {chunk}")
                        continue
            
        logger.info(f"vLLM API streaming response completed: {len(full_content)} chars")
        if full_reasoning:
            logger.info(f"Reasoning content: {len(full_reasoning)} chars")
        
        return full_content, full_reasoning, None  # 流式模式下通常没有logprobs
    except Exception as e:
        logger.error(f"Error in vLLM API streaming call: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = str(int(start_time * 1000))
    logger.info(f"[{request_id}] Request started: {request.method} {request.url}")
    logger.info(f"[{request_id}] Request headers: {request.headers}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"[{request_id}] Request completed: status_code={response.status_code}, time={process_time:.3f}s")
        return response
    except Exception as e:
        logger.error(f"[{request_id}] Request failed: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(e)}
        )

# Define request and response models
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    use_retrieval: Optional[bool] = False
    query_for_retrieval: Optional[str] = None
    retrieval_top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    response: str
    status: str
    retrieval_sources: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None  # Add field for CoT thinking
    logprobs: Optional[List[float]] = None
    
# Constants
DEFAULT_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Long live DeepSeek!
MAX_GPU_MEMORY = "13GiB"  # Adjust based on your GPU capacity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nyc_schools_data")

# Command line argument parsing 
parser = argparse.ArgumentParser(description="DeepSeek-R1 Model Server with RAG")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument("--data_dir", type=str, default=None, help="Directory containing RAG data")
parser.add_argument("--use_quantization", action="store_true", help="Use 4-bit quantization for efficiency")
parser.add_argument("--vllm_port", type=int, default=8000, help="Port where vLLM API is running")
args = parser.parse_args()

# If a data directory is specified, update related variables
if args.data_dir:
    DOCUMENTS_DIR = args.data_dir
    print(f"Using custom data directory: {DOCUMENTS_DIR}")
else:
    DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nyc_schools_data")
    print(f"Using default data directory: {DOCUMENTS_DIR}")

# Get port from command line arguments first, then environment, then default
port = args.port
if not port:
    # Try to get port from environment variable, with careful error handling
    try:
        port_env = os.environ.get("PORT", "8000")
        # Make sure the port is clean - strip any extra text
        port_env = port_env.strip().split("\n")[-1] if "\n" in port_env else port_env.strip()
        port = int(port_env)
    except (ValueError, TypeError):
        print("Error parsing PORT environment variable, using default 8000")
        port = 8000

print(f"Using port: {port}")

# Helper function to clean model output by removing special tokens and markers
def clean_output(text):
    # Remove common special tokens
    text = re.sub(r'<\|begin_of_sentence\|>', '', text)
    text = re.sub(r'<\|end_of_sentence\|>', '', text)
    text = re.sub(r'', '', text)
    text = re.sub(r'', '', text)
    
    # Remove DeepSeek role markers
    text = re.sub(r'<\|[a-zA-Z]+\|>', '', text)
    
    # Clean extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'^\s+', '', text)
    text = re.sub(r'\s+$', '', text)
    
    return text

# Extract thinking content if present (for CoT models)
def extract_thinking(text):
    thinking_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        # Remove the thinking section from the text
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text, thinking
    return text, None

vllm_model = None          # 全局句柄

def load_vllm_model():
    """Load DeepSeek-R1 with vLLM."""
    global vllm_model
    logger.info("Loading DeepSeek-R1 with vLLM …")
    t0 = time.time()

    vllm_model = LLM(
        model=DEFAULT_MODEL_PATH,
        dtype="float16",              # fp16
        # 可选: quantization="awq"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_PATH,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"vLLM loaded in {time.time()-t0:.1f}s")
    return vllm_model, tokenizer

# Load model
def load_model():
    logger.info(f"Loading DeepSeek-R1 model on {DEVICE}...")
    
    start_time = time.time()
    
    try:
        # Set quantization if needed to fit in GPU memory
        if DEVICE == "cuda":
            logger.info("Using CUDA with 4-bit quantization")
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
            logger.info("Using CPU (not recommended for performance)")
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
        logger.error(traceback.format_exc())
        raise e

def generate_response_vllm(messages,
                           tokenizer,
                           max_length=2048,
                           temperature=0.7,
                           top_p=0.9):
    """Generate text + token-level logprobs via vLLM."""
    # 1. prompt 拼接
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2. sampling 参数 – reason_outputs=True
    sampling_params = SamplingParams(
        max_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        reasoning_outputs=True,       # ⭐ 关键开关
    )

    # 3. 调用 vLLM
    outputs = vllm_model.generate(prompt, sampling_params)
    out = outputs[0].outputs[0]       # 仅取单条

    # 4. 解析
    raw_text = out.text
    response_clean = clean_output(raw_text)
    response_clean, thinking = extract_thinking(response_clean)

    return dict(
        text=response_clean.strip(),
        thinking=thinking,
        logprobs=out.logprobs          # 这里是 token 对齐的 logprobs 列表
    )

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
        logger.info(f"Generating response with params: max_length={max_length}, temp={temperature}, top_p={top_p}")
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                **generation_kwargs
            )
            
        # Decode output
        output = tokenizer.decode(generation_output[0], skip_special_tokens=False)
        
        # Add logging for raw output (truncated for readability)
        if len(output) > 500:
            logger.info(f"Raw output (truncated): {output[:250]}...{output[-250:]}")
        else:
            logger.info(f"Raw output: {output}")
        
        # Extract only the assistant response
        # For DeepSeek models, extract content between the last assistant marker and end of text
        response_start = output.rfind("<|assistant|>")
        if response_start != -1:
            response_start += len("<|assistant|>")
            response = output[response_start:]
        else:
            # Fallback extraction if marker not found
            response = output
        
        # Clean the response and extract thinking if present
        response = clean_output(response)
        response, thinking = extract_thinking(response)
        
        logger.info(f"Generated response length: {len(response)} chars")
        if thinking:
            logger.info(f"Extracted thinking: {thinking[:100]}...")
        
        return response.strip(), thinking
    except Exception as e:
        logger.error(f"Error occurred while generating response: {str(e)}")
        logger.exception(e)  # Log full traceback for debugging
        raise e

# Initialize FlashRAG retriever
def init_retriever(use_flashrag: bool = True):
    """Initialize the retriever with FlashRAG if available."""
    try:
        logger.info("Initializing FlashRAG retriever...")
        
        # Set up directories
        corpus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nyc_schools_data")
        processed_dir = os.path.join(corpus_dir, "processed")
        corpus_file = os.path.join(processed_dir, "corpus.jsonl")
        index_dir = os.path.join(processed_dir, "index")
        
        # Check for alternative index file (e5_Flat.index)
        alt_index_file = os.path.join(processed_dir, "e5_Flat.index")
        
        # Check if files exist
        index_exists = os.path.exists(index_dir)
        alt_index_exists = os.path.exists(alt_index_file)
        corpus_exists = os.path.exists(corpus_file)
        
        logger.info(f"Index exists: {index_exists}, Alt index exists: {alt_index_exists}, Corpus exists: {corpus_exists}")
        
        if use_flashrag:
            logger.info("Attempting to initialize retriever with FlashRAG...")
            
            if index_exists:
                # Use the regular index directory
                from rag.retriever import FlashRAGRetriever
                retriever = FlashRAGRetriever(
                    corpus_path=corpus_file,
                    index_path=index_dir,
                    model_name_or_path="intfloat/e5-base-v2"
                )
                logger.info("FlashRAG retriever initialized with standard index")
                return retriever
            elif alt_index_exists:
                # Create a symlink or copy the alternative index file
                try:
                    # Create index directory if it doesn't exist
                    if not os.path.exists(index_dir):
                        os.makedirs(index_dir)
                    
                    # Create a symlink from the alt index to the expected location
                    index_file_in_dir = os.path.join(index_dir, "index.faiss")
                    if not os.path.exists(index_file_in_dir):
                        logger.info(f"Creating symlink from {alt_index_file} to {index_file_in_dir}")
                        
                        # For Windows compatibility, use copy instead of symlink
                        import shutil
                        shutil.copy(alt_index_file, index_file_in_dir)
                        
                        # Also copy the corpus.jsonl to the index directory if needed
                        corpus_in_index = os.path.join(index_dir, "corpus.jsonl")
                        if not os.path.exists(corpus_in_index):
                            shutil.copy(corpus_file, corpus_in_index)
                    
                    # Now init with the index directory
                    from rag.retriever import FlashRAGRetriever
                    retriever = FlashRAGRetriever(
                        corpus_path=corpus_file,
                        index_path=index_dir,
                        model_name_or_path="intfloat/e5-base-v2"
                    )
                    logger.info("FlashRAG retriever initialized with converted alternative index")
                    return retriever
                except Exception as e:
                    logger.error(f"Failed to use alternative index: {str(e)}")
                    raise
    except Exception as e:
        logger.error(f"Failed to initialize FlashRAG retriever: {str(e)}")
        logger.warning("Falling back to basic retrieval without FlashRAG")
    
    # Fallback to regular retriever without FlashRAG
    from rag.retriever import BasicRetriever
    retriever = BasicRetriever(model_name_or_path="intfloat/e5-base-v2")
    
    # Load NYC schools data directly into the BasicRetriever
    if corpus_exists:
        try:
            logger.info(f"Loading corpus data from {corpus_file}")
            with open(corpus_file, 'r', encoding='utf-8') as f:
                corpus_data = [json.loads(line) for line in f]
            
            logger.info(f"Adding {len(corpus_data)} documents to BasicRetriever")
            retriever.add_documents(corpus_data)
        except Exception as e:
            logger.error(f"Error loading corpus data: {str(e)}")
    
    return retriever

# Application startup event
model = None
tokenizer = None
retriever = None

# Use lifespan context manager instead of on_event
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    global retriever
    logger.info("=== Starting model server ===")
    retriever = init_retriever()
    logger.info("Server initialization complete")
    yield
    logger.info("=== Shutting down model server ===")


app.router.lifespan_context = lifespan

# API endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    request_time = time.time()
    logger.info(f"Chat request received: {len(request.messages)} messages, use_retrieval={request.use_retrieval}")
    
    try:
        # Handle retrieval if requested
        if request.use_retrieval:
            query_for_retrieval = request.query_for_retrieval
            
            if not query_for_retrieval:
                for msg in reversed(request.messages):
                    if msg.get("role") == "user":
                        query_for_retrieval = msg.get("content", "")
                        break
            
            if query_for_retrieval:
                logger.info(f"Retrieval query: '{query_for_retrieval[:100]}...'")
                
                retrieval_start_time = time.time()
                retrieval_results = retriever.retrieve(
                    query_for_retrieval, 
                    top_k=request.retrieval_top_k
                )
                retrieval_time = time.time() - retrieval_start_time
                
                logger.info(f"Retrieved {len(retrieval_results)} documents in {retrieval_time:.2f}s")
                
                context = format_retrieved_documents(retrieval_results)
                
                has_system_msg = False
                augmented_messages = []
                
                for msg in request.messages:
                    if msg.get("role") == "system":
                        has_system_msg = True
                        msg["content"] = f"{msg['content']}\n\n{context}"
                    augmented_messages.append(msg)
                
                if not has_system_msg:
                    augmented_messages.insert(0, {
                        "role": "system",
                        "content": f"You are a helpful assistant. Use the following information to answer the user's question:\n\n{context}"
                    })
                
                generation_start_time = time.time()
                # Use streaming function instead
                response, reasoning_content, logprobs = await generate_response_vllm_api_stream(
                    augmented_messages,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p
                )
                generation_time = time.time() - generation_start_time
                
                logger.info(f"Generated response in {generation_time:.2f}s")
                
                total_time = time.time() - request_time
                logger.info(f"Total request time: {total_time:.2f}s")
                
                return ChatResponse(
                    response=response,
                    status="success",
                    retrieval_sources=retrieval_results,
                    thinking=reasoning_content,
                    logprobs=logprobs
                )
            else:
                logger.warning("Retrieval was requested but no query was provided or could be inferred.")
        
        # If not using retrieval, return regular response
        logger.info("Generating response without retrieval")
        generation_start_time = time.time()
        # Use streaming function instead
        response, reasoning_content, logprobs = await generate_response_vllm_api_stream(
            request.messages, 
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        generation_time = time.time() - generation_start_time
        
        logger.info(f"Generated response in {generation_time:.2f}s")
        
        total_time = time.time() - request_time
        logger.info(f"Total request time: {total_time:.2f}s")
        
        return ChatResponse(
            response=response, 
            status="success",
            thinking=reasoning_content,
            logprobs=logprobs
        )
    except Exception as e:
        logger.error(f"Error in /api/chat: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Add a streaming endpoint with Server-Sent Events
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    logger.info(f"Streaming chat request received: {len(request.messages)} messages")
    
    async def event_generator():
        try:
            # Initialize streaming payload
            payload = {
                "model": "DeepSeek-R1", 
                "messages": request.messages,
                "max_tokens": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": True
            }
            
            # Send the request to vLLM with streaming enabled
            async with httpx.AsyncClient(timeout=300.0, verify=False) as client:
                async with client.stream("POST", VLLM_SERVER_URL, json=payload) as response:
                    response.raise_for_status()
                    
                    # Stream each chunk back to the client as SSE
                    async for chunk in response.aiter_lines():
                        if not chunk.strip():
                            continue
                        
                        # Skip the "data: " prefix if it exists
                        if chunk.startswith("data: "):
                            chunk = chunk[6:]
                        
                        # Skip ending marker
                        if chunk == "[DONE]":
                            yield f"data: [DONE]\n\n"
                            break
                        
                        # Forward the chunk
                        yield f"data: {chunk}\n\n"
                        
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            error_json = json.dumps({"error": str(e)})
            yield f"data: {error_json}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.get("/api/health")
async def health_check():
    retrieval_ready = retriever is not None
    
    # 检查vLLM服务是否可用
    vllm_available = False
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
            response = await client.get("http://localhost:8000/v1/models")
            vllm_available = response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking vLLM availability: {str(e)}")
    
    logger.info(f"Health check: vllm_available={vllm_available}, retrieval_ready={retrieval_ready}, device={DEVICE}")
    
    status = "healthy" if vllm_available and retrieval_ready else "unavailable"
    
    return {
        "status": status, 
        "vllm_available": vllm_available,
        "device": DEVICE,
        "retrieval_ready": retrieval_ready,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/")
async def root():
    """Root endpoint for basic connection testing"""
    return {"message": "DeepSeek-R1 Model Server API is running. Use /api/chat for chat functionality."}

# Main function
if __name__ == "__main__":
    # Use the port we carefully determined at the top
    print(f"Starting uvicorn server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")