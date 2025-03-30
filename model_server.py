#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import uvicorn
import os
import logging
import time
import traceback
from typing import List, Dict, Any, Optional

# Import necessary libraries for LLaMA-3-8B
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import RAG components
from rag.retriever import DocumentRetriever, format_retrieved_documents

# Set environment variables at the top of the file to use scratch dir
os.environ["HF_HOME"] = "/scratch/sx2490/huggingface_cache"  # Use scratch directory for larger quota
os.makedirs("/scratch/sx2490/huggingface_cache", exist_ok=True)

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
app = FastAPI(title="Large Language Model API", description="Web API for large language model chat service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, this should be set to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = str(int(start_time * 1000))
    logger.info(f"[{request_id}] Request started: {request.method} {request.url}")
    
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
    
# Constants
DEFAULT_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # Long live DeepSeek!
MAX_GPU_MEMORY = "13GiB"  # Adjust based on your GPU capacity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nyc_schools_data")

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
        # Find the last assistant segment
        response_start = output.rfind("<|assistant|>") + len("<|assistant|>\n")
        response_end = output.rfind("</s>", response_start)
        
        # Add logging for response extraction
        logger.info(f"Response extraction: start={response_start}, end={response_end}")
        
        if response_end == -1:
            response = output[response_start:]
        else:
            response = output[response_start:response_end]
        
        logger.info(f"Generated response length: {len(response)} chars")
        return response.strip()
    except Exception as e:
        logger.error(f"Error occurred while generating response: {str(e)}")
        logger.exception(e)  # Log full traceback for debugging
        raise e

# Initialize FlashRAG retriever
def init_retriever():
    try:
        logger.info("Initializing FlashRAG retriever...")
        
        # Check processed data directory and index
        processed_dir = os.path.join(DOCUMENTS_DIR, "processed")
        if not os.path.exists(processed_dir):
            logger.warning(f"Processed documents directory {processed_dir} does not exist. Creating...")
            os.makedirs(processed_dir, exist_ok=True)
        
        # Check if index and corpus already exist before initializing retriever
        index_path = os.path.join(processed_dir, "index")
        corpus_path = os.path.join(processed_dir, "corpus.jsonl")
        
        index_exists = os.path.exists(index_path)
        corpus_exists = os.path.exists(corpus_path)
        logger.info(f"Index exists: {index_exists}, Corpus exists: {corpus_exists}")
        
        # Initialize retriever with fallback mechanism
        try:
            # First try with FlashRAG enabled
            logger.info("Attempting to initialize retriever with FlashRAG...")
            retriever = DocumentRetriever(
                embedding_model="intfloat/e5-base-v2",
                documents_dir=DOCUMENTS_DIR,
                index_path=index_path,
                use_flashrag=True
            )
            
            # Test retriever to ensure it's working
            if index_exists and corpus_exists:
                logger.info("Testing retriever with a sample query...")
                test_results = retriever.retrieve("test query", top_k=1)
                logger.info(f"Retriever test result count: {len(test_results)}")
                return retriever
                
        except Exception as e:
            logger.error(f"Error initializing FlashRAG retriever: {str(e)}")
            logger.error(traceback.format_exc())
        
        # If we get here, either the test failed or an exception occurred
        # Fall back to basic retriever
        logger.warning("Falling back to basic retrieval without FlashRAG")
        return DocumentRetriever(
            documents_dir=DOCUMENTS_DIR,
            use_flashrag=False
        )
    except Exception as e:
        logger.error(f"Fatal error initializing retriever: {str(e)}")
        logger.error(traceback.format_exc())
        # Create a dummy retriever that just returns empty results to prevent the server from crashing
        class DummyRetriever:
            def retrieve(self, *args, **kwargs):
                return []
            def batch_search(self, *args, **kwargs):
                return [[]]
        logger.warning("Using dummy retriever that returns empty results")
        return DummyRetriever()

# Application startup event
model = None
tokenizer = None
retriever = None

# Use lifespan context manager instead of on_event
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Startup: load model and initialize retriever
    global model, tokenizer, retriever
    try:
        logger.info("=== Starting model server ===")
        model, tokenizer = load_model()
        retriever = init_retriever()
        logger.info("Server initialization complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.error(traceback.format_exc())
    yield
    # Shutdown: cleanup
    logger.info("=== Shutting down model server ===")

app.router.lifespan_context = lifespan

# API endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    request_time = time.time()
    logger.info(f"Chat request received: {len(request.messages)} messages, use_retrieval={request.use_retrieval}")
    
    try:
        # Check if model is loaded
        if model is None or tokenizer is None:
            logger.error("Model or tokenizer not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
            
        # Log first and last user message for context (truncated for privacy)
        if request.messages:
            first_msg = request.messages[0].get('content', '')[:100]
            last_msg = request.messages[-1].get('content', '')[:100]
            logger.info(f"First message: '{first_msg}...'")
            logger.info(f"Last message: '{last_msg}...'")
        
        # Handle retrieval if requested
        if request.use_retrieval:
            # Determine query for retrieval
            query_for_retrieval = request.query_for_retrieval
            
            # If no retrieval query is specified, use the last user message
            if not query_for_retrieval:
                for msg in reversed(request.messages):
                    if msg.get("role") == "user":
                        query_for_retrieval = msg.get("content", "")
                        break
            
            if query_for_retrieval:
                logger.info(f"Retrieval query: '{query_for_retrieval[:100]}...'")
                
                # Get retrieval results
                retrieval_start_time = time.time()
                retrieval_results = retriever.retrieve(
                    query_for_retrieval, 
                    top_k=request.retrieval_top_k
                )
                retrieval_time = time.time() - retrieval_start_time
                
                logger.info(f"Retrieved {len(retrieval_results)} documents in {retrieval_time:.2f}s")
                
                # Format retrieval results as context
                context = format_retrieved_documents(retrieval_results)
                
                # Add system message as context
                has_system_msg = False
                augmented_messages = []
                
                for msg in request.messages:
                    if msg.get("role") == "system":
                        # If there's already a system message, add context to it
                        has_system_msg = True
                        msg["content"] = f"{msg['content']}\n\n{context}"
                    augmented_messages.append(msg)
                
                # If no system message exists, add a new one
                if not has_system_msg:
                    augmented_messages.insert(0, {
                        "role": "system",
                        "content": f"You are a helpful assistant. {context}"
                    })
                
                # Generate response with context-augmented messages
                generation_start_time = time.time()
                response = generate_response(
                    augmented_messages,
                    model,
                    tokenizer,
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
                    retrieval_sources=retrieval_results
                )
            else:
                # If retrieval query cannot be determined, fall back to regular response
                logger.warning("Retrieval was requested but no query was provided or could be inferred.")
                
        # If not using retrieval, return regular response
        logger.info("Generating response without retrieval")
        generation_start_time = time.time()
        response = generate_response(
            request.messages, 
            model, 
            tokenizer, 
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
            status="success"
        )
    except Exception as e:
        logger.error(f"Error in /api/chat: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    retrieval_ready = retriever is not None
    model_loaded = model is not None
    
    logger.info(f"Health check: model_loaded={model_loaded}, retrieval_ready={retrieval_ready}, device={DEVICE}")
    
    status = "healthy" if model_loaded else "unavailable"
    
    return {
        "status": status, 
        "model_loaded": model_loaded, 
        "device": DEVICE,
        "retrieval_ready": retrieval_ready,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/")
async def root():
    """Root endpoint for basic connection testing"""
    return {"message": "Model Server API is running. Use /api/chat for chat functionality."}

# Main function
if __name__ == "__main__":
    # Get port from environment variable, default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Start server
    logger.info(f"Starting uvicorn server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info") 