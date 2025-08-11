from datetime import datetime
import pandas as pd
import numpy as np
import re
import os
import json

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
        "wrong delivery address",
        "order not delivered",
        "late delivery"
    ],
    "Food stale": [
        "food was cold", 
        "biryani was too salty", 
        "burger was soggy", 
        "pizza was cold",
        "stale food",
        "not fresh",
        "poor food quality",
        "food was bad"
    ],
    "Delivery partner rude": [
        "delivery guy was rude", 
        "rider was rude", 
        "delivery person behaved badly", 
        "rider was impolite", 
        "rude to security guard",
        "unprofessional delivery person"
    ],
    "Maps not working properly": [
        "map location incorrect", 
        "maps not showing location properly",
        "location tracking issues",
        "wrong directions",
        "gps not working"
    ],
    "Instamart should be open all night": [
        "late-night instamart", 
        "instamart availability at night",
        "24/7 instamart service",
        "night delivery instamart",
        "midnight instamart"
    ],
    "Bring back 10 minute bolt delivery": [
        "bring back ten minute delivery", 
        "10-min delivery request",
        "miss bolt delivery",
        "want bolt delivery back",
        "request to restore 10 minute delivery"
    ],
    "Payment issues": [
        "cash on delivery not working",
        "payment failed",
        "payment method issues",
        "cod option",
        "unable to pay",
        "payment declined"
    ],
    "App issues": [
        "app crash", 
        "app slow", 
        "unable to update address",
        "app error",
        "login issues",
        "technical problems",
        "app not working"
    ],
    "High Charges/Fees": [
        "high delivery charges", 
        "extra fees", 
        "GST charges", 
        "platform fees",
        "expensive",
        "overcharged",
        "too costly"
    ],
    "Positive Feedback": [
        "good", 
        "nice", 
        "great", 
        "excellent", 
        "love", 
        "best", 
        "awesome",
        "amazing service",
        "good delivery",
        "fast delivery",
        "quick delivery",
        "prompt service"
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
        Tuple of (category_counts, categorized_reviews)
        where categorized_reviews is a dict mapping category names to lists of reviews
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
    
    # Initialize categorized reviews dictionary
    categorized_reviews = {cat: [] for cat in PREDEFINED_CATEGORIES.keys()}
    categorized_reviews["Other"] = []
    
    # If no reviews, return empty counts and categorized_reviews
    if not review_texts:
        return category_counts, categorized_reviews
    
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
        uncategorized_review_objects = []
        
        # Categorize each review
        for i, review_text in enumerate(review_texts):
            # Get the original review object
            original_review = reviews_list[i]
            
            # Search for similar examples - get top 3 matches
            results = vector_store.similarity_search(review_embeddings[i], k=3)
            
            if results and results[0][1] < 1.0:  # More lenient threshold to reduce "Other" category
                # Get category from metadata
                category = results[0][2].get("category", "Other")
                category_counts[category] += 1
                categorized_reviews[category].append(original_review)
            else:
                # Store for LLM processing
                uncategorized_reviews.append(review_text)
                uncategorized_review_objects.append(original_review)
        
        # Process uncategorized reviews with LLM if there are any
        if uncategorized_reviews:
            try:
                # Get new categories from LLM
                new_categories = suggest_new_category(
                    uncategorized_reviews, 
                    list(PREDEFINED_CATEGORIES.keys())
                )
                
                # Count reviews by new categories and store review objects
                for review_text, category in new_categories.items():
                    # Normalize category name to handle potential formatting issues
                    category = category.strip()
                    
                    if category not in category_counts:
                        category_counts[category] = 0
                        categorized_reviews[category] = []
                    
                    try:
                        # Find the corresponding review object
                        if review_text in uncategorized_reviews:
                            review_idx = uncategorized_reviews.index(review_text)
                            if review_idx >= 0 and review_idx < len(uncategorized_review_objects):
                                category_counts[category] += 1
                                categorized_reviews[category].append(uncategorized_review_objects[review_idx])
                        else:
                            # Try to find a close match
                            for i, text in enumerate(uncategorized_reviews):
                                if text.lower().strip() == review_text.lower().strip():
                                    category_counts[category] += 1
                                    categorized_reviews[category].append(uncategorized_review_objects[i])
                                    break
                    except Exception as e:
                        print(f"Error processing specific review for LLM categorization: {e}")
                        
                # Any remaining uncategorized reviews go to "Other"
                uncategorized_indices = set(range(len(uncategorized_reviews)))
                for review_text in new_categories.keys():
                    if review_text in uncategorized_reviews:
                        idx = uncategorized_reviews.index(review_text)
                        if idx in uncategorized_indices:
                            uncategorized_indices.remove(idx)
                
                # Add remaining reviews to "Other"
                remaining_reviews = []
                for idx in uncategorized_indices:
                    category_counts["Other"] += 1
                    categorized_reviews["Other"].append(uncategorized_review_objects[idx])
                    remaining_reviews.append(uncategorized_review_objects[idx])
                
                # If too many "Other" reviews, try one more time with a more aggressive approach
                if len(remaining_reviews) > 10:
                    print(f"Too many Other reviews ({len(remaining_reviews)}). Trying again with more aggressive categorization...")
                    
                    # Extract just the content from the reviews
                    remaining_texts = []
                    for review in remaining_reviews:
                        if isinstance(review, dict):
                            content = review.get('content', '')
                        else:
                            content = str(review)
                        
                        if content and not content.isspace():
                            remaining_texts.append(content)
                    
                    # Try to recategorize with a more aggressive prompt
                    try:
                        from app.llm_categorizer import last_resort_categorize
                        
                        # Get existing categories including any new ones that were created
                        all_categories = list(category_counts.keys())
                        all_categories.remove("Other")  # Remove "Other" from the list
                        
                        # Try more aggressive categorization
                        emergency_categories = last_resort_categorize(remaining_texts, all_categories)
                        
                        # Remove from "Other" and add to appropriate categories
                        for review_text, new_cat in emergency_categories.items():
                            # Find the review in the "Other" category
                            for i, review in enumerate(categorized_reviews["Other"]):
                                review_content = ""
                                if isinstance(review, dict):
                                    review_content = review.get('content', '')
                                else:
                                    review_content = str(review)
                                
                                if review_content.strip() == review_text.strip():
                                    # Remove from Other
                                    category_counts["Other"] -= 1
                                    removed_review = categorized_reviews["Other"].pop(i)
                                    
                                    # Add to new category
                                    if new_cat not in category_counts:
                                        category_counts[new_cat] = 0
                                        categorized_reviews[new_cat] = []
                                    
                                    category_counts[new_cat] += 1
                                    categorized_reviews[new_cat].append(removed_review)
                                    break
                        
                        print(f"Successfully recategorized reviews. Other category now has {category_counts['Other']} reviews.")
                    except Exception as e:
                        print(f"Error in emergency recategorization: {e}")
                    
            except Exception as e:
                print(f"Error in LLM categorization process: {e}")
                # If LLM categorization completely fails, try to assign reviews to existing categories
                # even with a very loose threshold
                
                # We already have the embeddings, so try with a much looser threshold
                for i, review_text in enumerate(uncategorized_reviews):
                    original_review = uncategorized_review_objects[i]
                    results = vector_store.similarity_search(review_embeddings[i], k=1)
                    
                    if results:  # Any match at all, no threshold
                        category = results[0][2].get("category", "Positive Feedback")  # Default to Positive Feedback instead of Other
                        category_counts[category] += 1
                        categorized_reviews[category].append(original_review)
                    else:
                        # Last resort, just put it in "Positive Feedback" if it has any positive words
                        review_lower = review_text.lower()
                        if any(word in review_lower for word in ["good", "great", "nice", "best", "love", "awesome", "excellent", "fast", "quick"]):
                            category_counts["Positive Feedback"] += 1
                            categorized_reviews["Positive Feedback"].append(original_review)
                        else:
                            # Truly last resort - "Other"
                            category_counts["Other"] += 1
                            categorized_reviews["Other"].append(original_review)
        
        return category_counts, categorized_reviews
        
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
        Tuple of (category_counts, categorized_reviews)
    """
    # Initialize category counts
    category_counts = {cat: 0 for cat in PREDEFINED_CATEGORIES.keys()}
    category_counts["Other"] = 0
    
    # Initialize categorized reviews dictionary
    categorized_reviews = {cat: [] for cat in PREDEFINED_CATEGORIES.keys()}
    categorized_reviews["Other"] = []
    
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
                    categorized_reviews[category].append(review)
                    matched = True
                    break
            if matched:
                break
                
        # If no category matched, count as "Other"
        if not matched:
            category_counts["Other"] += 1
            categorized_reviews["Other"].append(review)
    
    return category_counts, categorized_reviews

def load_reviews_from_csv(file_path):
    """Load reviews from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading reviews from {file_path}: {e}")
        return []

def get_category_counts_from_file(file_path):
    """
    Get category counts and categorized reviews from a CSV file
    
    Returns:
        Tuple of (category_counts, categorized_reviews)
    """
    reviews = load_reviews_from_csv(file_path)
    counts, categorized_reviews = categorize_reviews(reviews)
    
    # Convert to serializable format for JSON
    serializable_reviews = {}
    for category, reviews_list in categorized_reviews.items():
        serializable_reviews[category] = []
        for review in reviews_list:
            # Include only needed fields to keep response size smaller
            serializable_reviews[category].append({
                "content": review.get("content", ""),
                "score": review.get("score", ""),
                "userName": review.get("userName", ""),
                "at": str(review.get("at", ""))
            })
    
    return counts, serializable_reviews
