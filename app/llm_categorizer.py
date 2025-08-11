"""
LLM-based categorization for reviews that don't match existing categories
"""
import os
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

def last_resort_categorize(reviews, existing_categories):
    """
    Emergency categorization for reviews that were placed in the "Other" category
    Uses a more aggressive approach to ensure reviews are placed in meaningful categories
    
    Args:
        reviews (list): List of review texts that were previously uncategorized
        existing_categories (list): List of all available category names
        
    Returns:
        dict: Mapping of review texts to category names
    """
    if not client:
        print("OpenAI client not initialized")
        return {review: "Positive Feedback" for review in reviews}  # Default to Positive Feedback as last resort
    
    # Don't process if no reviews
    if not reviews:
        return {}
    
    # Limit the number of reviews to process (to avoid too large requests)
    sample_reviews = reviews[:100] if len(reviews) > 100 else reviews
    
    try:
        # Create a prompt for the LLM
        existing_categories_str = "\n".join([f"- {cat}" for cat in existing_categories])
        reviews_str = "\n".join([f"- {review}" for review in sample_reviews])
        
        prompt = f"""EMERGENCY CATEGORIZATION: I have a set of restaurant delivery app reviews that were placed in the "Other" category, but we need to assign them to more meaningful categories.
        
Available categories (YOU MUST USE THESE ONLY, NO NEW CATEGORIES ALLOWED):
{existing_categories_str}

Reviews to recategorize:
{reviews_str}

CRITICAL INSTRUCTIONS:
1. You MUST assign EVERY single review to one of the existing categories listed above
2. DO NOT create any new categories
3. DO NOT use "Other" category at all - this is the emergency pass to eliminate "Other"
4. When in doubt, use these default rules:
   - Any review with positive words → "Positive Feedback"
   - Any review mentioning delivery speed → "Positive Feedback" if positive, "Delivery issue" if negative
   - Any review about payment → "Payment Issues"
   - Any review about food quality → "Food stale"
   - Any review about app problems → "App issues"
   - Any review about fees → "High Charges/Fees"
   - Any review about service → "Positive Feedback" if positive tone
   - Any review with negative sentiment → find the closest category from the list
5. BE CREATIVE - stretch the meaning of categories if needed, but assign EVERY review
6. This is the FINAL categorization pass - if you don't categorize it here, it will remain as "Other"

Format your response as a simple JSON object with reviews as keys and categories as values like this:
{{
  "review text 1": "Category Name 1",
  "review text 2": "Category Name 2"
}}

REMEMBER: Use ONLY the categories listed above!
"""
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a specialist in categorizing problematic review texts that were hard to categorize."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse response
        result = response.choices[0].message.content
        import json
        try:
            categorized = json.loads(result)
            
            # Check if the response is in the expected format
            if not isinstance(categorized, dict):
                print(f"Unexpected response format from LLM: {result}")
                return {review: "Positive Feedback" for review in sample_reviews}  # Default to Positive Feedback
                
            # Validate the categories
            for review, category in list(categorized.items()):
                if category not in existing_categories:
                    # If category not in existing list, default to Positive Feedback
                    print(f"Invalid category '{category}' from LLM. Defaulting to Positive Feedback.")
                    categorized[review] = "Positive Feedback"
            
            # Make sure all reviews have a category
            for review in sample_reviews:
                if review not in categorized:
                    categorized[review] = "Positive Feedback"  # Default to Positive Feedback
            
            return categorized
            
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {result}")
            return {review: "Positive Feedback" for review in sample_reviews}  # Default to Positive Feedback
        
    except Exception as e:
        print(f"Error in emergency categorization: {e}")
        # Fallback to Positive Feedback category - better than "Other"
        return {review: "Positive Feedback" for review in sample_reviews}

def suggest_new_category(reviews, existing_categories):
    """
    Use LLM to suggest a new category for reviews that don't match existing ones
    
    Args:
        reviews (list): List of review texts that didn't match existing categories
        existing_categories (list): List of existing category names
        
    Returns:
        dict: Mapping of review texts to suggested category names
    """
    if not client:
        print("OpenAI client not initialized")
        return {"Other": reviews}
    
    # Don't process if no reviews
    if not reviews:
        return {}
    
    # Limit the number of reviews to process (to avoid too large requests)
    sample_reviews = reviews[:50] if len(reviews) > 50 else reviews
    
    try:
        # Create a prompt for the LLM
        existing_categories_str = "\n".join([f"- {cat}" for cat in existing_categories])
        reviews_str = "\n".join([f"- {review}" for review in sample_reviews])
        
        prompt = f"""I have a set of restaurant delivery app reviews that need to be categorized.
        
Existing categories:
{existing_categories_str}

Uncategorized reviews:
{reviews_str}

IMPORTANT: Every review MUST be assigned to EITHER an existing category OR a new meaningful category. DO NOT use the "Other" category unless absolutely impossible to categorize.

For each review, please:
1. Try your best to fit it into one of the existing categories listed above
2. If it really doesn't match any existing category, create a NEW specific category name that clearly describes the issue
3. For general positive comments about service or delivery, use "Positive Feedback"
4. For payment method issues like "cash on delivery", use "Payment Issues"
5. For complaints about delivery time promises, create an appropriate category

Critical guidelines:
- "good delivery" or "fast delivery" belongs in "Positive Feedback"
- Any review mentioning payment, cash on delivery, or COD belongs in "Payment Issues"
- Reviews about app functionality belong in "App issues"
- ANY review with positive sentiment should be in "Positive Feedback"
- Be creative in finding ways to categorize every review
- Avoid using "Other" category at all costs - it should be your absolute last resort
- If you must use "Other", limit it to no more than 5% of reviews

Format your response as a JSON object like this:
{{
  "new_categories": [
    {{
      "name": "Category Name 1",
      "reviews": ["review text 1", "review text 2", ...] 
    }},
    ...
  ],
  "other": ["review text that is absolutely impossible to categorize", ...]
}}
"""
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes app reviews."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse response
        result = response.choices[0].message.content
        import json
        categorized = json.loads(result)
        
        # Create mapping of reviews to categories
        review_categories = {}
        
        # Process new categories
        if "new_categories" in categorized:
            for category in categorized["new_categories"]:
                cat_name = category["name"]
                for review in category["reviews"]:
                    if review in sample_reviews:
                        review_categories[review] = cat_name
        
        # Process "other" category
        if "other" in categorized:
            for review in categorized["other"]:
                if review in sample_reviews:
                    review_categories[review] = "Other"
        
        # Default any unprocessed reviews to "Other"
        for review in sample_reviews:
            if review not in review_categories:
                review_categories[review] = "Other"
                
        return review_categories
        
    except Exception as e:
        print(f"Error using LLM to suggest categories: {e}")
        # Fallback to "Other" category
        return {"Other": sample_reviews}
