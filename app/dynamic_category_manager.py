"""
Dynamic category management for InsiteGent
This module handles loading, saving, and tracking dynamically created categories
"""
import os
import json
from datetime import datetime

# Path to the dynamic categories file
CATEGORIES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dynamic_categories.json")

# Ensure data directory exists
os.makedirs(os.path.dirname(CATEGORIES_FILE), exist_ok=True)

def load_dynamic_categories():
    """
    Load dynamically created categories from JSON storage
    
    Returns:
        dict: Dictionary mapping category names to example phrases
    """
    if not os.path.exists(CATEGORIES_FILE):
        return {}
    
    try:
        with open(CATEGORIES_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dynamic categories: {e}")
        return {}

def save_dynamic_categories(categories):
    """
    Save dynamically created categories to JSON storage
    
    Args:
        categories (dict): Dictionary mapping category names to example phrases
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(CATEGORIES_FILE), exist_ok=True)
        
        with open(CATEGORIES_FILE, "w") as f:
            json.dump(categories, f, indent=2)
            
        print(f"Saved {len(categories)} dynamic categories")
    except Exception as e:
        print(f"Error saving dynamic categories: {e}")

def add_dynamic_category(category_name, example_phrases=None):
    """
    Add a new dynamic category or update an existing one
    
    Args:
        category_name (str): Name of the category
        example_phrases (list): List of example phrases for this category
        
    Returns:
        bool: True if successful, False otherwise
    """
    if example_phrases is None:
        example_phrases = []
        
    try:
        # Load existing categories
        categories = load_dynamic_categories()
        
        # Add or update the category
        if category_name in categories:
            # Add new examples, avoid duplicates
            existing_examples = set(categories[category_name])
            for phrase in example_phrases:
                if phrase and phrase not in existing_examples:
                    categories[category_name].append(phrase)
        else:
            # Create new category
            categories[category_name] = example_phrases
            print(f"Created new dynamic category: {category_name}")
        
        # Save updated categories
        save_dynamic_categories(categories)
        return True
    except Exception as e:
        print(f"Error adding dynamic category: {e}")
        return False

def get_all_categories():
    """
    Get all categories, both predefined and dynamic
    
    Returns:
        dict: Dictionary mapping category names to example phrases
    """
    from app.categorizer import PREDEFINED_CATEGORIES
    
    # Start with predefined categories
    all_categories = {**PREDEFINED_CATEGORIES}
    
    # Add dynamic categories
    dynamic_categories = load_dynamic_categories()
    all_categories.update(dynamic_categories)
    
    return all_categories

def is_existing_category(category_name):
    """
    Check if a category already exists (predefined or dynamic)
    
    Args:
        category_name (str): Name of the category to check
        
    Returns:
        bool: True if category exists, False otherwise
    """
    all_categories = get_all_categories()
    return category_name in all_categories
