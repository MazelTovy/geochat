#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Initialize RAG components for the GeoChat system.
This script processes documents and builds the retrieval index for RAG.
"""

import os
import sys
import argparse
import logging
import time
import gc
import traceback
import psutil

from rag.document_processor import DocumentProcessor
from rag.retriever import DocumentRetriever

# Configure logging with enhanced detail and file output
os.makedirs("/scratch/sx2490/Logs/model_server", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/scratch/sx2490/Logs/model_server/init_rag.log')
    ]
)
logger = logging.getLogger("init_rag")

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("Checking dependencies")
    
    dependencies = {
        "numpy": "For numerical computing",
        "torch": "PyTorch for deep learning",
        "transformers": "Hugging Face Transformers for language models",
        "datasets": "Hugging Face Datasets for data handling",
        "faiss-cpu": "Facebook AI Similarity Search for vector search",
        "psutil": "For system monitoring",
        "einops": "For tensor operations",
        "tiktoken": "For tokenization",
        "scipy": "For scientific computing"
    }
    
    missing = []
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} - {description}")
        except ImportError:
            logger.warning(f"✗ {package} - {description} - NOT FOUND")
            missing.append(package)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        
        # Suggest pip install command
        command = f"pip install {' '.join(missing)}"
        logger.info(f"To install missing dependencies, run: {command}")
        return False
    
    logger.info("All dependencies are installed")
    return True

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB (RSS)")
    logger.info(f"Virtual memory: {memory_info.vms / (1024 * 1024):.2f} MB (VMS)")
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    logger.info(f"System memory: {system_memory.available / (1024 * 1024):.2f} MB available out of {system_memory.total / (1024 * 1024):.2f} MB total")
    logger.info(f"System memory usage: {system_memory.percent}%")

def set_environment_variables():
    """Set environment variables for better memory management"""
    logger.info("Setting environment variables for memory management")
    
    # Set cache directories to scratch
    scratch_dir = "/scratch/sx2490"
    
    # Create directories
    cache_dirs = {
        "HF_HOME": f"{scratch_dir}/huggingface_cache",
        "TRANSFORMERS_CACHE": f"{scratch_dir}/transformers_cache",
        "HF_DATASETS_CACHE": f"{scratch_dir}/datasets_cache",
        "TORCH_HOME": f"{scratch_dir}/torch_cache",
    }
    
    for var_name, dir_path in cache_dirs.items():
        os.environ[var_name] = dir_path
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Set {var_name} to {dir_path}")
    
    # Set PyTorch memory management options
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    logger.info("Environment variables set successfully")

def main():
    logger.info("=== RAG Initialization Script ===")
    
    parser = argparse.ArgumentParser(description='Initialize RAG system by processing documents and building index')
    parser.add_argument('--docs-dir', type=str, default='nyc_schools_data', 
                        help='Directory containing documents to process')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory to save processed documents and index')
    parser.add_argument('--chunk-size', type=int, default=500, 
                        help='Size of document chunks')
    parser.add_argument('--chunk-overlap', type=int, default=50, 
                        help='Overlap between consecutive chunks')
    parser.add_argument('--embedding-model', type=str, default='intfloat/e5-base-v2', 
                        help='Model to use for document embeddings')
    parser.add_argument('--use-flashrag', action='store_true', default=True,
                        help='Use FlashRAG for retrieval')
    parser.add_argument('--check-only', action='store_true', 
                        help='Only check dependencies without initializing RAG')
    
    args = parser.parse_args()
    
    try:
        # Check dependencies
        if not check_dependencies():
            if args.check_only:
                logger.info("Dependency check failed, exiting as requested")
                return 1
            else:
                logger.warning("Dependency check failed, but continuing with initialization")
        elif args.check_only:
            logger.info("Dependency check passed, exiting as requested")
            return 0
        
        # Set environment variables for better memory management
        set_environment_variables()
        
        # Log initial memory usage
        logger.info("Initial memory usage:")
        log_memory_usage()
        
        # Resolve paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        docs_dir = os.path.join(script_dir, args.docs_dir)
        output_dir = args.output_dir or os.path.join(docs_dir, 'processed')
        
        logger.info(f"Using documents from: {docs_dir}")
        logger.info(f"Processed documents will be saved to: {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Process documents
        logger.info("=== Step 1: Processing documents ===")
        start_time = time.time()
        
        processor = DocumentProcessor(
            documents_dir=docs_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            output_dir=output_dir
        )
        
        # List available documents
        docs = processor.list_documents()
        if not docs:
            logger.warning(f"No documents found in {docs_dir}. Please add documents before processing.")
            return 1
            
        logger.info(f"Found {len(docs)} documents to process.")
        
        # Process all documents
        processed_docs = processor.process_all_documents()
        logger.info(f"Processed {len(processed_docs)} documents into chunks.")
        
        process_time = time.time() - start_time
        logger.info(f"Document processing completed in {process_time:.2f} seconds.")
        
        # Log memory usage after document processing
        logger.info("Memory usage after document processing:")
        log_memory_usage()
        
        # Step 2: Build index with FlashRAG
        logger.info("=== Step 2: Building index with FlashRAG ===")
        start_time = time.time()
        
        if args.use_flashrag:
            try:
                # Initialize retriever with FlashRAG
                logger.info("Creating retriever with FlashRAG enabled")
                retriever = DocumentRetriever(
                    embedding_model=args.embedding_model,
                    documents_dir=docs_dir,
                    index_path=os.path.join(output_dir, "index"),
                    use_flashrag=True
                )
                
                # Build index
                logger.info("Building index - this may take some time depending on document size")
                retriever.build_index()
                logger.info("FlashRAG index built successfully.")
                
                # Log memory usage after building index
                logger.info("Memory usage after building index:")
                log_memory_usage()
                
                # Test the retriever with a sample query
                logger.info("Testing retriever with a sample query...")
                sample_query = "Tell me about the schools in New York City"
                
                test_start_time = time.time()
                results = retriever.retrieve(sample_query, top_k=3)
                test_time = time.time() - test_start_time
                
                logger.info(f"Sample query: '{sample_query}'")
                logger.info(f"Retrieved {len(results)} documents in {test_time:.2f} seconds")
                
                for i, doc in enumerate(results):
                    logger.info(f"Document {i+1}: {doc.get('title', 'Unknown Title')}")
                    logger.info(f"Score: {doc.get('score', 0)}")
                    logger.info(f"Source: {doc.get('source', 'Unknown Source')}")
                    
                    content = doc.get('content', '')
                    preview = content[:100] + "..." if len(content) > 100 else content
                    logger.info(f"Content preview: {preview}")
                    logger.info("---")
                
            except Exception as e:
                logger.error(f"Failed to build index with FlashRAG: {str(e)}")
                logger.error(traceback.format_exc())
                logger.error("Falling back to basic retrieval.")
                
                # Free memory before retrying
                gc.collect()
                log_memory_usage()
                
                # Initialize basic retriever
                retriever = DocumentRetriever(
                    embedding_model=args.embedding_model,
                    documents_dir=docs_dir,
                    use_flashrag=False
                )
        else:
            logger.info("FlashRAG disabled, using basic retrieval.")
            # Initialize basic retriever
            retriever = DocumentRetriever(
                embedding_model=args.embedding_model,
                documents_dir=docs_dir,
                use_flashrag=False
            )
        
        index_time = time.time() - start_time
        logger.info(f"Index building completed in {index_time:.2f} seconds.")
        
        # Final memory usage
        logger.info("Final memory usage:")
        log_memory_usage()
        
        logger.info("=== RAG initialization complete ===")
        logger.info("You can now use the RAG system with the processed documents.")
        
        return 0
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 