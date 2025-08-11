from datetime import datetime
import pandas as pd
import numpy as np
import re
import os

# Import our custom modules
from app.embedding_utils import get_embeddings, get_text_embedding
from app.vector_store import VectorStore
from app.llm_categorizer import suggest_new_category

# Define predefined categories with example phrases
PREDEFINED_CATEGORIES = {
    "Delivery issue": [
        "order arrived late", 
        "delivery delay", 
        "rider got lost", 
        "didn't follow instructions",
        "delivery took too long",
        "wrong delivery address"
    ],
    "Food stale": [
        "food was cold", 
        "biryani was too salty", 
        "burger was soggy", 
        "pizza was cold",
        "stale food",
        "not fresh"
    ],
    "Delivery partner rude": [
        "delivery guy was rude", 
        "rider was rude", 
        "delivery person behaved badly", 
        "rider was impolite", 
        "rude to security guard"
    ],
    "Maps not working properly": [
        "map location incorrect", 
        "maps not showing location properly",
        "location tracking issues",
        "wrong directions"
    ],
    "Instamart should be open all night": [
        "late-night instamart", 
        "instamart availability at night",
        "24/7 instamart service",
        "night delivery"
    ],
    "Bring back 10 minute bolt delivery": [
        "bring back ten minute delivery", 
        "10-min delivery request",
        "fast delivery option",
        "quick delivery"
    ],
    "App issues": [
        "app crash", 
        "app slow", 
        "unable to update address",
        "payment failed",
        "login issues"
    ],
    "High Charges/Fees": [
        "high delivery charges", 
        "extra fees", 
        "GST charges", 
        "platform fees",
        "expensive",
        "overcharged"
    ],
    "Positive Feedback": [
        "good", 
        "nice", 
        "great", 
        "excellent", 
        "love", 
        "best", 
        "awesome",
        "amazing service"
    ]
}

def categorize_reviews(reviews_list):
    """
    Categorize app reviews into categories using vector similarity
    
    Uses OpenAI embeddings and FAISS vector store for semantic matching.
    Falls back to LLM categorization for reviews that don't match existing categories.
    
    Args:
        reviews_list: List of review dictionaries or DataFrame rows 
    
    Returns:
        Dictionary with categories and counts
    """
    # Extract review texts
    review_texts = []
    for review in reviews_list:
        if isinstance(review, dict):
            content = review.get('content', '')
        else:
            content = str(review.get('content', ''))
        
        if content and not content.isspace():
            review_texts.append(content)
    
    # Initialize results
    category_counts = {cat: 0 for cat in PREDEFINED_CATEGORIES.keys()}
    category_counts["Other"] = 0
    
    # If no reviews, return empty counts
    if not review_texts:
        return category_counts
    
    # Try vector-based categorization
    try:
        # Prepare examples and category mappings
        examples = []
        example_categories = []
        
        for category, phrases in PREDEFINED_CATEGORIES.items():
            for phrase in phrases:
                examples.append(phrase)
                example_categories.append(category)
        
        # Get embeddings for examples and reviews
        example_embeddings = get_embeddings(examples)
        review_embeddings = get_embeddings(review_texts)
        
        if example_embeddings is None or review_embeddings is None:
            # If embeddings fail, fall back to regex matching
            return fallback_categorize_reviews(reviews_list)
        
        # Set up vector store
        vector_store = VectorStore(dimension=len(example_embeddings[0]))
        
        # Add examples to vector store with category metadata
        metadata = [{"category": cat} for cat in example_categories]
        vector_store.add_texts(examples, example_embeddings, metadata)
        
        # Initialize storage for uncategorized reviews
        uncategorized_reviews = []
        
        # Categorize each review
        for i, review in enumerate(review_texts):
            # Search for similar examples
            results = vector_store.similarity_search(review_embeddings[i], k=1)
            
            if results and results[0][1] < 1.2:  # Similarity threshold
                # Get category from metadata
                category = results[0][2].get("category", "Other")
                category_counts[category] += 1
            else:
                # Store for LLM processing
                uncategorized_reviews.append(review)
        
        # Process uncategorized reviews with LLM if there are any
        if uncategorized_reviews:
            # Get new categories from LLM
            new_categories = suggest_new_category(
                uncategorized_reviews, 
                list(PREDEFINED_CATEGORIES.keys())
            )
            
            # Count reviews by new categories
            for review, category in new_categories.items():
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1
        
        return category_counts
        
    except Exception as e:
        print(f"Error in vector categorization: {e}")
        # Fall back to regex matching
        return fallback_categorize_reviews(reviews_list)

def fallback_categorize_reviews(reviews_list):
    """
    Fallback categorization using regex pattern matching when vector approach fails
    
    Args:
        reviews_list: List of review dictionaries or DataFrame rows 
    
    Returns:
        Dictionary with categories and counts
    """
    # Initialize category counts
    category_counts = {cat: 0 for cat in PREDEFINED_CATEGORIES.keys()}
    category_counts["Other"] = 0
    
    # Process each review
    for review in reviews_list:
        # Get review content
        if isinstance(review, dict):
            content = review.get('content', '').lower()
        else:
            content = str(review.get('content', '')).lower()
        
        if not content:
            continue
            
        # Check for matches
        matched = False
        for category, patterns in PREDEFINED_CATEGORIES.items():
            for pattern in patterns:
                if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', content, re.IGNORECASE):
                    category_counts[category] += 1
                    matched = True
                    break
            if matched:
                break
                
        # If no category matched, count as "Other"
        if not matched:
            category_counts["Other"] += 1
    
    return category_counts

def load_reviews_from_csv(file_path):
    """Load reviews from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading reviews from {file_path}: {e}")
        return []

def get_category_counts_from_file(file_path):
    """Get category counts from a CSV file"""
    reviews = load_reviews_from_csv(file_path)
    return categorize_reviews(reviews)
