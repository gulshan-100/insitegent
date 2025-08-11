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
        
        prompt = f"""I have a set of restaurant delivery app reviews that don't match my existing categories.
        
Existing categories:
{existing_categories_str}

Uncategorized reviews:
{reviews_str}

Please analyze these reviews and create 1-3 new meaningful categories that capture their main themes. Then, assign each review to either one of your new categories or to "Other" if it doesn't fit well.

Format your response as a JSON object like this:
{{
  "new_categories": [
    {{
      "name": "Category Name 1",
      "reviews": ["review text 1", "review text 2", ...] 
    }},
    ...
  ],
  "other": ["review text that doesn't fit", ...]
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
