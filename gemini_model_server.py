#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import logging
import time
import traceback
import json
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import argparse
import asyncio

# Import RAG components
from rag.simple_retriever import SimpleRetriever

# Set up logging
os.makedirs("/scratch/sx2490/Logs/model_server", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/scratch/sx2490/Logs/model_server/gemini_server.log')
    ]
)
logger = logging.getLogger("gemini_model_server")

# Create FastAPI application
app = FastAPI(title="Gemini Chat API", description="Web API for Gemini chat service with RAG capabilities")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Request and response models
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
    timestamp: str = None

# Initialize Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")  # 请在这里填入您的 API Key
if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
    logger.warning("GEMINI_API_KEY not set! Please set it in environment variables or update the code.")
    
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

# Parse command line arguments
parser = argparse.ArgumentParser(description="Gemini Model Server with RAG")
parser.add_argument("--port", type=int, default=8001, help="Port to run the server on")
parser.add_argument("--data_dir", type=str, default=None, help="Directory containing RAG data")
args = parser.parse_args()

# Set up data directory
if args.data_dir:
    DOCUMENTS_DIR = args.data_dir
else:
    DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nyc_schools_data")

logger.info(f"Using data directory: {DOCUMENTS_DIR}")

# Initialize retriever
retriever = None

def init_retriever():
    """Initialize the simple retriever with NYC schools data"""
    global retriever
    try:
        retriever = SimpleRetriever(documents_dir=DOCUMENTS_DIR)
        retriever.load_documents()
        logger.info("Simple retriever initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {str(e)}")
        retriever = None

# Initialize retriever on startup
init_retriever()

# School recommendation system prompt
SCHOOL_RECOMMENDATION_PROMPT = """
You are a helpful NYC school recommendation assistant. When users ask for school recommendations, you must format your response in a very specific way that allows the map to display school markers correctly.

CRITICAL: You MUST use the exact school data format provided in the RAG context. Never make up school codes or addresses.

IMPORTANT OUTPUT FORMAT REQUIREMENTS:
For each recommended school, you MUST use this EXACT format structure:

### 1. **School Name (School Code)**
- **Location:** Full street address from the data, Borough, NY zipcode
- **Key Features:** List the most important features, programs, and characteristics
- **Seats Remaining:** [program name]: [number] seats left (use exact seat availability data from RAG context)
- **Why It Fits:** Explain why this school matches the user's needs

### 2. **School Name (School Code)**
- **Location:** Full street address from the data, Borough, NY zipcode  
- **Key Features:** List the most important features, programs, and characteristics
- **Seats Remaining:** [program name]: [number] seats left (use exact seat availability data from RAG context)
- **Why It Fits:** Explain why this school matches the user's needs

CRITICAL FORMAT RULES - FOLLOW EXACTLY:
1. Each school entry MUST start with "### [number]. **School Name (School Code)**"
2. School names MUST be in bold with double asterisks: **School Name**
3. School codes MUST be taken EXACTLY from the RAG data (like "01M140", "02M167", "03M333") and placed in parentheses
4. Location MUST be the EXACT address from the RAG data, including street name, borough, and zip code
5. Use "- **Location:**", "- **Key Features:**", "- **Seats Remaining:**", and "- **Why It Fits:**" exactly as shown
6. NEVER make up school codes or addresses - only use data from the RAG context
7. MANDATORY: Include "- **Seats Remaining:**" section with exact seat availability data from RAG context

REQUIRED INFORMATION TO INCLUDE:
- School name and exact location from RAG data
- School code exactly as it appears in the data
- Grade levels served
- Enrollment numbers
- Special programs and features
- Admission methods
- Available seats information (MANDATORY): Always include seat availability status
  * If seat data is available: show "Available Seats: [program]: [seats_left] seats left"
  * If no seat data found: show "Available Seats: No current seat availability data found"
- Performance metrics if relevant

EXAMPLE FORMAT:
### 1. **P.S. 333 Manhattan School for Children (03M333)**
- **Location:** 333 West 17 Street, Manhattan, NY 10011
- **Key Features:** Grades PK-8, enrollment 482, diverse student body, 30% students with IEPs, dual language programs available
- **Seats Remaining:** General Education: 15 seats left
- **Why It Fits:** This school offers excellent diversity and has a strong academic program with support for students with special needs

Remember: The map component depends on this EXACT format to parse schools and display them as markers. Any deviation will prevent the map from working. ALWAYS use the exact school data from the RAG context - never invent information.
"""

def format_messages_for_gemini(messages: List[Dict[str, str]]) -> str:
    """Format messages for Gemini API"""
    formatted = []
    
    # Check if this appears to be a school recommendation request
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "").lower()
            break
    
    is_school_request = any(keyword in user_message for keyword in [
        "school", "schools", "recommend", "suggestion", "education", "primary", "elementary", "middle", "high school"
    ])
    
    # Add system prompt for school recommendations
    if is_school_request:
        formatted.append(f"System Instructions: {SCHOOL_RECOMMENDATION_PROMPT}")
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            formatted.append(f"Additional Instructions: {content}")
        elif role == "assistant":
            formatted.append(f"Assistant: {content}")
        else:
            formatted.append(f"User: {content}")
    return "\n\n".join(formatted)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests"""
    try:
        logger.info(f"Received chat request with {len(request.messages)} messages")
        
        # Get the latest user message
        user_message = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Prepare context
        context = ""
        
        # Check if this is a school-related request
        is_school_request = any(keyword in user_message.lower() for keyword in [
            "school", "schools", "recommend", "suggestion", "education", "primary", "elementary", "middle", "high school", "central park"
        ])
        
        # Perform retrieval if requested OR if it's a school-related request
        if (request.use_retrieval or is_school_request) and retriever:
            query = request.query_for_retrieval or user_message
            logger.info(f"Performing retrieval for query: {query[:50]}... (auto-enabled: {is_school_request})")
            
            retrieved_docs = retriever.retrieve(query, top_k=request.retrieval_top_k or 10)
            
            if retrieved_docs:
                context = "\n\nHere is relevant school information from the NYC Schools database:\n\n"
                for i, doc in enumerate(retrieved_docs):
                    context += f"School {i+1}:\n{doc['content']}\n\n"
                    # Include raw data for school recommendations
                    if 'data' in doc and is_school_request:
                        school_data = doc['data']
                        if 'basic_info' in school_data:
                            basic_info = school_data['basic_info']
                            location = basic_info.get('location', {})
                            if location:
                                context += f"Full Address: {location.get('full_address', 'Address not available')}\n"
                        context += "---\n\n"
        
        # Format messages for Gemini
        prompt = format_messages_for_gemini(request.messages)
        
        # Add context if available
        if context:
            prompt = context + "\n\nConversation:\n" + prompt
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_output_tokens": request.max_length,
            }
        )
        
        # Extract response text
        response_text = response.text
        
        return ChatResponse(
            response=response_text,
            status="success",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return ChatResponse(
            response=f"Sorry, an error occurred: {str(e)}",
            status="error",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Handle streaming chat requests"""
    async def event_generator():
        try:
            # Similar processing as non-streaming
            user_message = ""
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            # Check if this is a school-related request
            is_school_request = any(keyword in user_message.lower() for keyword in [
                "school", "schools", "recommend", "suggestion", "education", "primary", "elementary", "middle", "high school", "central park"
            ])
            
            context = ""
            if (request.use_retrieval or is_school_request) and retriever:
                query = request.query_for_retrieval or user_message
                retrieved_docs = retriever.retrieve(query, top_k=request.retrieval_top_k or 10)
                
                if retrieved_docs:
                    context = "\n\nHere is relevant school information from the NYC Schools database:\n\n"
                    for i, doc in enumerate(retrieved_docs):
                        context += f"School {i+1}:\n{doc['content']}\n\n"
                        # Include raw data for school recommendations
                        if 'data' in doc and is_school_request:
                            school_data = doc['data']
                            if 'basic_info' in school_data:
                                basic_info = school_data['basic_info']
                                location = basic_info.get('location', {})
                                if location:
                                    context += f"Full Address: {location.get('full_address', 'Address not available')}\n"
                            context += "---\n\n"
            
            prompt = format_messages_for_gemini(request.messages)
            if context:
                prompt = context + "\n\nConversation:\n" + prompt
            
            # Generate streaming response
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_output_tokens": request.max_length,
                },
                stream=True
            )
            
            # Stream the response
            for chunk in response:
                if chunk.text:
                    data = json.dumps({
                        "choices": [{
                            "delta": {"content": chunk.text},
                            "index": 0,
                            "finish_reason": None
                        }]
                    })
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
            
            # Send final message
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            error_data = json.dumps({
                "error": str(e),
                "status": "error"
            })
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "gemini-1.5-flash",
        "retriever": "initialized" if retriever else "not initialized",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Gemini Chat API Server",
        "version": "1.0",
        "endpoints": [
            "/api/chat - Chat endpoint",
            "/api/chat/stream - Streaming chat endpoint",
            "/api/health - Health check"
        ]
    }

if __name__ == "__main__":
    port = args.port
    logger.info(f"Starting Gemini server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 