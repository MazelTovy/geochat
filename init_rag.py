#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG Initialization Script
This script processes documents and builds the retrieval index for RAG.
"""

import os
import argparse
import logging
import sys
from rag.document_processor import DocumentProcessor
from rag.retriever import DocumentRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("init_rag")

def main():
    parser = argparse.ArgumentParser(description='Initialize RAG system by processing documents and building index')
    parser.add_argument('--docs-dir', type=str, default='documents', 
                        help='Directory containing documents to process')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory to save processed documents and index')
    parser.add_argument('--chunk-size', type=int, default=500, 
                        help='Size of document chunks')
    parser.add_argument('--chunk-overlap', type=int, default=50, 
                        help='Overlap between consecutive chunks')
    parser.add_argument('--embedding-model', type=str, default='all-mpnet-base-v2', 
                        help='Model to use for document embeddings')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(script_dir, args.docs_dir)
    output_dir = args.output_dir or os.path.join(docs_dir, 'processed')
    
    logger.info(f"Using documents from: {docs_dir}")
    logger.info(f"Processed documents will be saved to: {output_dir}")
    
    # Step 1: Process documents
    logger.info("=== Step 1: Processing documents ===")
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
        return
        
    logger.info(f"Found {len(docs)} documents to process.")
    
    # Process all documents
    processed_docs = processor.process_all_documents()
    logger.info(f"Processed {len(processed_docs)} documents into chunks.")
    
    # Step 2: Build index (placeholder for now)
    logger.info("=== Step 2: Building index ===")
    logger.info("This step will be implemented in a future update.")
    logger.info("For now, the system will use the processed document chunks directly.")
    
    logger.info("=== RAG initialization complete ===")
    logger.info("You can now use the RAG system with the processed documents.")
    
if __name__ == "__main__":
    main() 