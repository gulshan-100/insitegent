"""
Utility functions for generating embeddings using OpenAI
"""
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = None
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")

def get_embeddings(texts):
    """
    Get embeddings for a list of texts using OpenAI's API
    
    Args:
        texts (list): List of text strings to embed
    
    Returns:
        np.ndarray: Array of embeddings or None if failed
    """
    if not client:
        print("OpenAI client not initialized")
        return None
    
    try:
        # Handle empty input
        if not texts or len(texts) == 0:
            return None
            
        # Make API call to get embeddings
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        
        # Extract and convert embeddings to numpy array
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def get_text_embedding(text):
    """
    Get embedding for a single text string
    
    Args:
        text (str): Text to embed
    
    Returns:
        np.ndarray: Embedding vector or None if failed
    """
    embeddings = get_embeddings([text])
    if embeddings is not None and len(embeddings) > 0:
        return embeddings[0]
    return None
