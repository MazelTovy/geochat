"""
Document Retriever for RAG (Retrieval Augmented Generation)
This module handles the retrieval of relevant documents based on the query.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import sys
import json
import torch
import faiss
from sentence_transformers import SentenceTransformer

# Add FlashRAG path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import FlashRAG components if available
try:
    from flashrag.config import Config
    from flashrag.utils import get_retriever as get_flashrag_retriever
    from flashrag.prompt import PromptTemplate
    FLASHRAG_AVAILABLE = True
except ImportError:
    FLASHRAG_AVAILABLE = False

logger = logging.getLogger("retriever")

class DocumentRetriever:
    """
    Base retriever class for accessing and retrieving relevant documents.
    This integrates with FlashRAG for high-performance retrieval.
    """
    
    def __init__(self, 
                 embedding_model: str = "intfloat/e5-base-v2",
                 documents_dir: str = "documents",
                 index_path: Optional[str] = None,
                 top_k: int = 5,
                 use_flashrag: bool = True):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: The model to use for embedding documents and queries
            documents_dir: Directory containing documents
            index_path: Path to pre-built index, if available
            top_k: Number of documents to retrieve by default
            use_flashrag: Whether to use FlashRAG for retrieval
        """
        self.embedding_model_name = embedding_model
        self.documents_dir = documents_dir
        self.index_path = index_path
        self.top_k = top_k
        self.use_flashrag = use_flashrag
        self.flashrag_retriever = None
        self.documents = []
        self.document_embeddings = None
        self._retrieval_cache = {}
        
        if self.use_flashrag:
            self._init_flashrag()
        
        logger.info(f"Retriever initialized with {embedding_model} model.")
        
    def _init_flashrag(self):
        """
        Initialize FlashRAG retriever if available.
        """
        if not FLASHRAG_AVAILABLE:
            logger.warning("FlashRAG is not available. Please install it to use advanced retrieval capabilities.")
            self.use_flashrag = False
            return

        try:
            # Build FlashRAG configuration
            config_dict = {
                "save_note": "nyc_schools",
                "model2path": {"e5": self.embedding_model_name},
                "retrieval_method": "e5",
                "corpus_path": os.path.join(self.documents_dir, "processed/corpus.jsonl"),
                "index_path": self.index_path or os.path.join(self.documents_dir, "processed/index"),
                "retrieval_topk": self.top_k,
                "save_retrieval_cache": True
            }
            
            self.config = Config(config_dict=config_dict)
            
            # Check if index needs to be built first
            index_dir = os.path.dirname(self.config.index_path)
            os.makedirs(index_dir, exist_ok=True)
            
            # If index doesn't exist, create an empty corpus file first
            corpus_path = self.config.corpus_path
            corpus_dir = os.path.dirname(corpus_path)
            os.makedirs(corpus_dir, exist_ok=True)
            
            # Initialize FlashRAG retriever
            self.flashrag_retriever = get_flashrag_retriever(self.config)
            
            logger.info(f"FlashRAG retriever initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize FlashRAG retriever: {str(e)}")
            self.use_flashrag = False
    
    def _prepare_corpus(self):
        """
        Prepare document corpus for FlashRAG in the required format.
        """
        try:
            # Processed documents directory
            processed_dir = os.path.join(self.documents_dir, "processed")
            if not os.path.exists(processed_dir):
                logger.warning(f"Processed documents directory {processed_dir} does not exist.")
                return
            
            # Read all processed documents
            corpus_data = []
            for filename in os.listdir(processed_dir):
                if filename.endswith('.json') and filename != 'corpus.jsonl':
                    file_path = os.path.join(processed_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            chunks = json.load(f)
                            for chunk in chunks:
                                corpus_data.append({
                                    "id": chunk["id"],
                                    "title": chunk["doc_id"],
                                    "contents": chunk["content"],
                                    "metadata": {"source": chunk["source"]}
                                })
                    except Exception as e:
                        logger.error(f"Error loading document {file_path}: {str(e)}")
            
            # Write to corpus.jsonl
            corpus_path = os.path.join(processed_dir, "corpus.jsonl")
            with open(corpus_path, 'w', encoding='utf-8') as f:
                for item in corpus_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
            logger.info(f"Prepared corpus with {len(corpus_data)} document chunks at {corpus_path}")
            
            # Update corpus path in configuration
            self.config.corpus_path = corpus_path
            
            # Reinitialize retriever to use the new corpus
            self.flashrag_retriever = get_flashrag_retriever(self.config)
            
            return corpus_path
        except Exception as e:
            logger.error(f"Error preparing corpus: {str(e)}")
            return None
        
    def build_index(self):
        """
        Build the search index for fast retrieval.
        """
        if not self.use_flashrag:
            logger.warning("FlashRAG not available, skipping index building.")
            return
        
        try:
            # Prepare corpus
            corpus_path = self._prepare_corpus()
            if not corpus_path:
                return
            
            # Use FlashRAG's index building functionality
            logger.info(f"Building index at {self.config.index_path}")
            self.flashrag_retriever._build_index()
            logger.info("Index built successfully.")
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents for the query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of dictionaries containing document info and relevance scores
        """
        if top_k is None:
            top_k = self.top_k
            
        # Check cache first
        cache_key = f"{query}_{top_k}"
        if cache_key in self._retrieval_cache:
            logger.info(f"Retrieved from cache for query: {query[:30]}...")
            return self._retrieval_cache[cache_key]
        
        logger.info(f"Retrieval requested for query: {query[:30]}... (top_k={top_k})")
        
        if self.use_flashrag and self.flashrag_retriever:
            try:
                # Use FlashRAG for retrieval
                results = self.flashrag_retriever.search(query, num=top_k)
                
                # Convert results to standard format
                formatted_results = []
                for i, result in enumerate(results):
                    formatted_results.append({
                        "id": result.get("id", f"doc{i}"),
                        "content": result.get("contents", ""),
                        "score": result.get("score", 1.0 - (i/top_k)),
                        "source": result.get("metadata", {}).get("source", f"source_{i}.txt"),
                        "title": result.get("title", "")
                    })
                
                # Cache results
                self._retrieval_cache[cache_key] = formatted_results
                
                return formatted_results
            except Exception as e:
                logger.error(f"Error using FlashRAG retriever: {str(e)}")
                # If FlashRAG fails, fall back to dummy implementation
        
        # If FlashRAG is not available or fails, return dummy results
        dummy_results = [
            {
                "id": f"doc{i}",
                "content": f"This is placeholder content for document {i}. Implement actual retrieval logic.",
                "score": (top_k - i) / top_k,
                "source": f"source_{i}.txt"
            }
            for i in range(top_k)
        ]
        
        return dummy_results
    
    def batch_search(self, queries: List[str], top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Perform batch retrieval for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of documents to retrieve per query
            
        Returns:
            List of retrieval results for each query
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, top_k))
        return results
    
    def _save_cache(self):
        """
        Save retrieval cache to disk.
        """
        cache_path = os.path.join(self.documents_dir, "processed/retrieval_cache.json")
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._retrieval_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved retrieval cache with {len(self._retrieval_cache)} entries to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving retrieval cache: {str(e)}")


class FlashRAGRetriever:
    """
    A wrapper class for the FlashRAG retriever that ensures compatibility with the model server.
    Uses the dense retrieval approach from FlashRAG.
    """
    
    def __init__(self, corpus_path: str, index_path: str, model_name_or_path: str = "intfloat/e5-base-v2"):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.model_name = model_name_or_path
        
        # Load corpus
        self.corpus = self._load_corpus()
        
        # Initialize the index if it exists
        if os.path.exists(index_path):
            if os.path.isdir(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
                self.index = faiss.read_index(os.path.join(index_path, "index.faiss"))
            else:
                logger.warning(f"Index directory {index_path} exists but does not contain index.faiss")
                self.index = None
        else:
            logger.warning(f"Index path {index_path} does not exist")
            self.index = None
            
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name_or_path)
        logger.info(f"FlashRAGRetriever initialized with {model_name_or_path}")
        
    def _load_corpus(self):
        """Load the corpus from the JSONL file"""
        corpus = []
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        corpus.append(item)
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing JSON line: {line}")
                        continue
            logger.info(f"Loaded corpus with {len(corpus)} documents")
            return corpus
        except Exception as e:
            logger.error(f"Failed to load corpus from {self.corpus_path}: {str(e)}")
            return []
            
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Legacy API compatibility method"""
        return self.search(query, num=top_k)
        
    def search(self, query: str, num: int = 5) -> List[Dict[str, Any]]:
        """
        Search the corpus for documents relevant to the query
        
        Args:
            query: The search query
            num: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing document info
        """
        if self.index is None:
            logger.error("Index is not initialized. Cannot perform search.")
            return []
            
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1).astype(np.float32)
        
        # Search the index
        scores, indices = self.index.search(query_embedding_np, k=num)
        
        # Get the documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.corpus):
                doc = self.corpus[idx]
                doc["score"] = float(scores[0][i])
                results.append(doc)
                
        return results


class BasicRetriever:
    """
    A simple retriever that uses sentence-transformers for embedding and semantic search
    """
    
    def __init__(self, model_name_or_path: str = "intfloat/e5-base-v2"):
        self.model_name = model_name_or_path
        self.model = SentenceTransformer(model_name_or_path)
        self.corpus = []
        self.corpus_embeddings = None
        logger.info(f"BasicRetriever initialized with {model_name_or_path}")
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the retriever
        
        Args:
            documents: List of document dictionaries
        """
        self.corpus.extend(documents)
        texts = [doc.get("contents", "") or doc.get("content", "") for doc in documents]
        
        # Create embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        if self.corpus_embeddings is None:
            self.corpus_embeddings = embeddings
        else:
            self.corpus_embeddings = torch.cat([self.corpus_embeddings, embeddings])
            
        logger.info(f"Added {len(documents)} documents to BasicRetriever. Total: {len(self.corpus)}")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Legacy API compatibility method"""
        return self.search(query, num=top_k)
        
    def search(self, query: str, num: int = 5) -> List[Dict[str, Any]]:
        """
        Search the corpus for documents relevant to the query
        
        Args:
            query: The search query
            num: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing document info
        """
        if len(self.corpus) == 0:
            logger.warning("No documents in the corpus. Cannot perform search.")
            return []
            
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarities
        cos_scores = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), self.corpus_embeddings)
        
        # Get top-k results
        top_k_results = []
        top_results = torch.topk(cos_scores, min(num, len(self.corpus)))
        
        for score, idx in zip(top_results[0], top_results[1]):
            doc = self.corpus[idx]
            doc_copy = doc.copy()  # Create a copy to avoid modifying the original
            doc_copy["score"] = float(score)
            top_k_results.append(doc_copy)
            
        return top_k_results


# Factory function to get the appropriate retriever
def get_retriever(retriever_type: str = "flashrag", **kwargs) -> DocumentRetriever:
    """
    Factory function to create a retriever based on the specified type.
    
    Args:
        retriever_type: Type of retriever to create
        **kwargs: Additional arguments for the retriever
        
    Returns:
        An instance of DocumentRetriever
    """
    # Prioritize using FlashRAG as the default retriever
    if retriever_type == "flashrag":
        return DocumentRetriever(use_flashrag=True, **kwargs)
    elif retriever_type == "default" or retriever_type == "basic":
        return DocumentRetriever(use_flashrag=False, **kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

# Function to format retrieved documents for the model
def format_retrieved_documents(documents: List[Dict[str, Any]]) -> str:
    """
    Format retrieved documents into a string that can be included in the model prompt.
    
    Args:
        documents: List of retrieved documents
        
    Returns:
        Formatted string containing retrieved information
    """
    if not documents:
        return ""
    
    formatted_text = "Here is some relevant information that might help answer the question:\n\n"
    
    for i, doc in enumerate(documents):
        formatted_text += f"Document {i+1}:\n"
        formatted_text += f"{doc.get('content', doc.get('contents', ''))}\n"
        if 'title' in doc and doc['title']:
            formatted_text += f"Title: {doc['title']}\n"
        formatted_text += f"Source: {doc.get('source', doc.get('id', 'Unknown'))}\n\n"
    
    formatted_text += "Please use this information to help answer the user's question."
    
    return formatted_text 