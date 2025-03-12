"""
RAG Manager for integrating retrieval with the LLM generation process.
This module coordinates the document retrieval and incorporation into prompts.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import time

from .retriever import get_retriever, format_retrieved_documents
from .document_processor import DocumentProcessor

logger = logging.getLogger("rag_manager")

class RAGManager:
    """
    Manager class for Retrieval Augmented Generation.
    Coordinates document processing, retrieval, and integration with LLM.
    """
    
    def __init__(self, 
                 documents_dir: str = "documents",
                 retriever_type: str = "default",
                 embedding_model: str = "all-mpnet-base-v2",
                 index_path: Optional[str] = None):
        """
        Initialize the RAG manager.
        
        Args:
            documents_dir: Directory containing documents
            retriever_type: Type of retriever to use
            embedding_model: Model to use for embeddings
            index_path: Path to pre-built index, if available
        """
        self.documents_dir = documents_dir
        
        # Initialize the document processor
        self.doc_processor = DocumentProcessor(documents_dir=documents_dir)
        
        # Initialize the retriever
        self.retriever = get_retriever(
            retriever_type=retriever_type,
            embedding_model=embedding_model,
            documents_dir=documents_dir,
            index_path=index_path
        )
        
        logger.info(f"RAG Manager initialized")
        
    def process_documents(self):
        """
        Process all documents in the documents directory.
        """
        return self.doc_processor.process_all_documents()
        
    def build_index(self):
        """
        Build the retrieval index from processed documents.
        To be implemented when retriever is completed.
        """
        logger.info("Index building placeholder. Not yet implemented.")
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        start_time = time.time()
        results = self.retriever.retrieve(query, top_k=top_k)
        retrieval_time = time.time() - start_time
        
        logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.2f} seconds")
        return results
        
    def augment_query(self, query: str, messages: List[Dict[str, str]], top_k: int = 3) -> List[Dict[str, str]]:
        """
        Augment the conversation with retrieved information.
        
        Args:
            query: The query to use for retrieval
            messages: The current conversation messages
            top_k: Number of documents to retrieve
            
        Returns:
            Augmented conversation messages
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=top_k)
        
        if not retrieved_docs:
            logger.info("No relevant documents found for query")
            return messages
            
        # Format retrieved documents
        context = format_retrieved_documents(retrieved_docs)
        
        # Create a new system message with retrieved content
        # Or append to an existing system message
        has_system = False
        for i, message in enumerate(messages):
            if message.get("role") == "system":
                has_system = True
                # Append retrieval results to existing system message
                messages[i]["content"] = f"{messages[i]['content']}\n\n{context}"
                break
                
        if not has_system:
            # Add a new system message with retrieval results
            messages.insert(0, {
                "role": "system",
                "content": context
            })
            
        logger.info(f"Query augmented with {len(retrieved_docs)} retrieved documents")
        return messages, retrieved_docs
        
    def save_feedback(self, query: str, doc_ids: List[str], relevance_scores: List[float]):
        """
        Save user feedback for retrieved documents to improve retrieval.
        To be implemented.
        
        Args:
            query: The query
            doc_ids: List of document IDs
            relevance_scores: User relevance scores for each document
        """
        logger.info("Feedback saving placeholder. Not yet implemented.") 