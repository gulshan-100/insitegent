"""
Test script for dynamic category creation
"""
import os
import sys
import json
from datetime import datetime
import pandas as pd

# Add parent directory to path so we can import app modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import app modules
from app.categorizer import categorize_reviews
from app.dynamic_category_manager import get_all_categories

def load_test_reviews():
    """Load some sample reviews for testing"""
    # Use first CSV file from swiggy_reviews folder for test data
    csv_files = [f for f in os.listdir(os.path.join(parent_dir, 'swiggy_reviews')) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in swiggy_reviews folder")
        return []
    
    # Load the first file
    sample_path = os.path.join(parent_dir, 'swiggy_reviews', csv_files[0])
    try:
        df = pd.read_csv(sample_path)
        # Convert to list of dictionaries
        reviews = df.to_dict('records')
        return reviews[:20]  # Just use 20 reviews for testing
    except Exception as e:
        print(f"Error loading sample reviews: {e}")
        return []

def test_dynamic_categorization():
    """Test the dynamic categorization system"""
    print("Loading test reviews...")
    reviews = load_test_reviews()
    
    if not reviews:
        print("No test reviews available.")
        return
    
    print(f"Loaded {len(reviews)} test reviews")
    
    # Display the first few reviews
    print("\nSample reviews:")
    for i, review in enumerate(reviews[:5]):
        content = review.get('content', '')
        print(f"{i+1}. {content[:100]}...")
    
    # Get all categories before categorization
    print("\nExisting categories:")
    before_categories = get_all_categories()
    for cat in sorted(before_categories.keys()):
        print(f"- {cat}")
    
    # Categorize the reviews
    print("\nCategorizing reviews...")
    category_counts, categorized_reviews = categorize_reviews(reviews)
    
    # Display the categorization results
    print("\nCategorization results:")
    for category, count in category_counts.items():
        if count > 0:
            print(f"{category}: {count} reviews")
            # Show a sample review from this category
            if categorized_reviews[category]:
                review = categorized_reviews[category][0]
                content = review.get('content', '')
                print(f"  Sample: {content[:100]}...")
    
    # Check for new categories
    print("\nChecking for new categories...")
    after_categories = get_all_categories()
    new_categories = set(after_categories.keys()) - set(before_categories.keys())
    
    if new_categories:
        print(f"Created {len(new_categories)} new categories:")
        for cat in sorted(new_categories):
            print(f"- {cat}")
            # Show examples for this category
            examples = after_categories.get(cat, [])
            for example in examples[:2]:
                print(f"  Example: {example[:100]}...")
    else:
        print("No new categories were created.")

if __name__ == "__main__":
    print("Testing dynamic category creation")
    print("=" * 50)
    test_dynamic_categorization()
    print("=" * 50)
    print("Test complete")
