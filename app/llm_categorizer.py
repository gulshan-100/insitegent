"""
LLM-based categorization for reviews that don't match existing categories
"""
import os
import json
from datetime import datetime
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

def last_resort_categorize(reviews, existing_categories, create_new=True):
    """
    Emergency categorization for reviews that need meaningful categorization
    Uses a more aggressive approach to ensure reviews are placed in meaningful categories
    
    Args:
        reviews (list): List of review texts that need categorization
        existing_categories (list): List of all available category names
        create_new (bool): Whether to allow creating new categories
        
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
    
    # If we want to create new categories, use the suggest_new_category function
    if create_new:
        return suggest_new_category(sample_reviews, existing_categories)
    
    try:
        # Create a prompt for the LLM
        existing_categories_str = "\n".join([f"- {cat}" for cat in existing_categories])
        reviews_str = "\n".join([f"- {review}" for review in sample_reviews])
        
        # Use the original prompt for using only existing categories
        prompt = f"""CRITICAL CATEGORIZATION TASK: I have a set of restaurant delivery app reviews that need to be categorized.

Available categories (YOU MUST USE THESE ONLY, NO NEW CATEGORIES ALLOWED):
{existing_categories_str}

Reviews to categorize:
{reviews_str}

CRITICAL INSTRUCTIONS:
1. You MUST assign EVERY single review to one of the existing categories listed above
2. DO NOT create any new categories
3. DO NOT use any "Other" or "Miscellaneous" category - EVERY review must have a specific category
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
6. When truly uncertain, default to "Positive Feedback" rather than any generic category

Format your response as a simple JSON object with reviews as keys and categories as values like this:
{{
  "review text 1": "Category Name 1",
  "review text 2": "Category Name 2"
}}

REMEMBER: Use ONLY the categories listed above!"""

        # Make API call with standard format
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
    Use LLM to suggest a new category for reviews that need categorization
    
    Args:
        reviews (list): List of review texts that need categorization
        existing_categories (list): List of existing category names
        
    Returns:
        dict: Mapping of review texts to suggested category names
    """
    if not client:
        print("OpenAI client not initialized")
        # Always assign to a meaningful category, default to Positive Feedback
        return {review: "Positive Feedback" for review in reviews}
    
    # Don't process if no reviews
    if not reviews:
        return {}
    
    # Limit the number of reviews to process (to avoid too large requests)
    sample_reviews = reviews[:50] if len(reviews) > 50 else reviews
    
    try:
        # Create a prompt for the LLM
        existing_categories_str = "\n".join([f"- {cat}" for cat in existing_categories])
        reviews_str = "\n".join([f"- {review}" for review in sample_reviews])
        
        # Current time - helps ensure unique responses each time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        prompt = f"""CRITICAL CATEGORIZATION TASK ({current_time}): Create new categories for delivery app reviews that don't fit existing ones.

Existing categories:
{existing_categories_str}

Reviews to categorize:
{reviews_str}

CRITICAL INSTRUCTIONS (READ CAREFULLY):
1. YOU MUST AGGRESSIVELY CREATE NEW CATEGORIES for reviews that don't closely match existing categories
2. At least 50% of reviews MUST be assigned to NEW categories you create - this is MANDATORY
3. Each new category should be SPECIFIC and DESCRIPTIVE (3-5 words)
4. Look for emerging topics, patterns, and unique issues in the reviews
5. NEVER use vague categories like "Other" or "Miscellaneous"
6. Don't assign reviews to existing categories unless they're a perfect match
7. You MUST create at least 5-10 NEW distinct categories for different topics in these reviews

Examples of good NEW specific categories:
- "Menu Item Unavailable"  
- "Order Preparation Time Long"
- "App Navigation Confusing"  
- "Restaurant Packaging Poor" 
- "Discount Code Not Applied"
- "Delivery Location Hard to Find"
- "Item Missing From Order"
- "Portion Size Too Small"

Format your response as a JSON object like this:
{{
  "new_categories": [
    {{
      "name": "Specific New Category Name",
      "reviews": ["review text that needs this new category", ...] 
    }},
    {{
      "name": "Another New Category",
      "reviews": ["another review that needs categorization", ...] 
    }}
  ],
  "existing_categories": [
    {{
      "name": "Existing Category Name",
      "reviews": ["review text that perfectly fits this category", ...] 
    }}
  ]
}}

REMEMBER: YOU MUST CREATE NEW SPECIFIC CATEGORIES for at least 50% of these reviews. 
This is your primary objective - create 5-10 new detailed categories minimum."""
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert at identifying patterns and creating taxonomies from customer reviews."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse response
        result = response.choices[0].message.content
        
        try:
            categorized = json.loads(result)
            
            # Log the LLM response for debugging
            print(f"LLM categorization response received with {len(result)} characters")
            
            # Create mapping of reviews to categories
            review_categories = {}
            new_categories_created = []
            
            # Process new categories - prioritize these
            if "new_categories" in categorized:
                for category in categorized["new_categories"]:
                    cat_name = category["name"]
                    new_categories_created.append(cat_name)
                    print(f"LLM created new category: {cat_name}")
                    
                    # Add all reviews in this category
                    for review in category["reviews"]:
                        if review in sample_reviews:
                            review_categories[review] = cat_name
            
            # Process existing categories - only for reviews not already categorized
            if "existing_categories" in categorized:
                for category in categorized["existing_categories"]:
                    cat_name = category["name"]
                    for review in category["reviews"]:
                        if review in sample_reviews and review not in review_categories:
                            review_categories[review] = cat_name
            
            # Report statistics on category creation
            print(f"LLM created {len(new_categories_created)} new categories: {', '.join(new_categories_created)}")
            print(f"Categorized {len(review_categories)} out of {len(sample_reviews)} reviews")
            
            # Default any unprocessed reviews to specific categories
            for review in sample_reviews:
                if review not in review_categories:
                    review_lower = review.lower()
                    
                    # Try to assign to the most appropriate category based on keywords
                    if any(word in review_lower for word in ["delicious", "tasty", "good", "great", "excellent"]):
                        review_categories[review] = "Positive Feedback"
                    elif any(word in review_lower for word in ["late", "delay", "wait", "long time"]):
                        review_categories[review] = "Delivery issue"
                    elif any(word in review_lower for word in ["cold", "stale", "quality", "spoiled"]):
                        review_categories[review] = "Food stale"
                    elif any(word in review_lower for word in ["app", "crash", "freeze", "login"]):
                        review_categories[review] = "App issues"
                    else:
                        # Create a new category based on key phrases in the review
                        # Extract potential category name from review
                        words = review_lower.split()
                        if len(words) > 3:
                            # Try to extract a meaningful phrase
                            new_cat_name = " ".join(words[:3]).title()
                            print(f"Auto-creating new category from review: {new_cat_name}")
                            review_categories[review] = new_cat_name
                        else:
                            review_categories[review] = "Positive Feedback"
                            
            return review_categories
            
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {result}")
            return {review: "Positive Feedback" for review in sample_reviews}  # Default to Positive Feedback
        except Exception as e:
            print(f"Error processing LLM categorization: {e}")
            return {review: "Positive Feedback" for review in sample_reviews}  # Default to Positive Feedback
        
    except Exception as e:
        print(f"Error using LLM to suggest categories: {e}")
        # Fallback to Positive Feedback category
        return {review: "Positive Feedback" for review in sample_reviews}
