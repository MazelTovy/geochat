"""
Document Processor for RAG (Retrieval Augmented Generation)
This module handles the processing of documents: loading, chunking, and indexing.
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Callable
import json

logger = logging.getLogger("document_processor")

class DocumentProcessor:
    """
    Process documents for retrieval: load, clean, chunk, and prepare for indexing.
    """
    
    def __init__(self, 
                 documents_dir: str = "documents",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 output_dir: Optional[str] = None):
        """
        Initialize the document processor.
        
        Args:
            documents_dir: Directory containing documents to process
            chunk_size: Size of document chunks in tokens/chars
            chunk_overlap: Overlap between consecutive chunks
            output_dir: Directory to save processed documents
        """
        self.documents_dir = documents_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.output_dir = output_dir or os.path.join(documents_dir, "processed")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Document processor initialized with chunk size {chunk_size} and overlap {chunk_overlap}")
    
    def list_documents(self, file_extensions: List[str] = ['.txt', '.md', '.pdf', '.docx']) -> List[str]:
        """
        List all documents in the documents directory with specified extensions.
        
        Args:
            file_extensions: List of file extensions to include
            
        Returns:
            List of file paths
        """
        all_files = []
        
        if not os.path.exists(self.documents_dir):
            logger.warning(f"Documents directory {self.documents_dir} does not exist.")
            return all_files
            
        for root, _, files in os.walk(self.documents_dir):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    all_files.append(os.path.join(root, file))
                    
        logger.info(f"Found {len(all_files)} documents in {self.documents_dir}")
        return all_files
    
    def load_document(self, file_path: str) -> str:
        """
        Load document content from file.
        This is a simple implementation for text files.
        Extensions for PDF, DOCX, etc. to be added.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document content as string
        """
        logger.info(f"Loading document: {file_path}")
        
        try:
            if file_path.endswith('.txt') or file_path.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.endswith('.pdf'):
                logger.warning("PDF support not yet implemented. Returning empty string.")
                return ""
            elif file_path.endswith('.docx'):
                logger.warning("DOCX support not yet implemented. Returning empty string.")
                return ""
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return ""
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return ""
    
    def chunk_document(self, text: str) -> List[str]:
        """
        Split document into overlapping chunks.
        
        Args:
            text: Document text
            
        Returns:
            List of document chunks
        """
        if not text:
            return []
            
        # Simple character-based chunking for now
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence ending within the last 100 chars of the chunk
                search_area = text[max(end-100, start):end]
                sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', search_area)]
                
                if sentence_ends:
                    # Adjust end to the last sentence boundary found
                    end = max(end-100, start) + sentence_ends[-1]
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            
        logger.info(f"Document split into {len(chunks)} chunks")
        return chunks
    
    def process_all_documents(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all documents in the documents directory.
        
        Returns:
            Dictionary mapping document IDs to lists of chunks
        """
        all_docs = self.list_documents()
        processed_docs = {}
        
        for doc_path in all_docs:
            doc_id = os.path.basename(doc_path)
            doc_text = self.load_document(doc_path)
            chunks = self.chunk_document(doc_text)
            
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    "id": f"{doc_id}_chunk_{i}",
                    "doc_id": doc_id,
                    "content": chunk,
                    "chunk_index": i,
                    "source": doc_path
                })
            
            processed_docs[doc_id] = processed_chunks
            
            # Save processed chunks
            output_path = os.path.join(self.output_dir, f"{doc_id}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
                
        logger.info(f"Processed {len(processed_docs)} documents")
        return processed_docs 