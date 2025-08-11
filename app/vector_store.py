"""
Vector store implementation using FAISS
"""
import numpy as np
import faiss

class VectorStore:
    """
    A simple vector store implementation using FAISS for efficient similarity search
    """
    
    def __init__(self, dimension=1536):
        """
        Initialize a vector store with the specified dimension
        
        Args:
            dimension (int): The dimension of the vectors to be stored (default: 1536 for OpenAI embeddings)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.metadata = []
    
    def add_texts(self, texts, embeddings=None, metadata=None):
        """
        Add texts and their embeddings to the vector store
        
        Args:
            texts (list): List of text strings
            embeddings (np.ndarray, optional): Pre-computed embeddings
            metadata (list, optional): List of metadata dictionaries for each text
            
        Returns:
            list: Indices of the added texts
        """
        if not texts:
            return []
            
        if embeddings is None:
            # If no embeddings provided, return without adding
            return []
            
        # Convert embeddings to float32 (required by FAISS)
        embeddings = np.array(embeddings).astype(np.float32)
        
        # Add embeddings to the index
        self.index.add(embeddings)
        
        # Store texts and metadata
        start_idx = len(self.texts)
        self.texts.extend(texts)
        
        # Add metadata if provided
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])
            
        # Return indices of added texts
        return list(range(start_idx, len(self.texts)))
    
    def similarity_search(self, query_embedding, k=5):
        """
        Find the k most similar texts to the query
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of results to return
            
        Returns:
            list: List of tuples (text, score, metadata)
        """
        if self.index.ntotal == 0:
            return []
            
        # Prepare query embedding
        query_embedding = np.array([query_embedding]).astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append((
                    self.texts[idx],
                    float(distances[0][i]),  # Convert to Python float for easier serialization
                    self.metadata[idx]
                ))
                
        return results
