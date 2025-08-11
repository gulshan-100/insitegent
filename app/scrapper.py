from google_play_scraper import reviews, Sort
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json

def scrape_reviews(app_id, max_reviews):
    """
    Scrape reviews of input app id and return it as a list
    """
    try:
        # Get reviews from Google Play Store
        result, continuation_token = reviews(
            app_id,
            lang='en',
            country='in',
            sort=Sort.NEWEST,
            count=max_reviews
        )
        
        # Save reviews to data folder for future use
        try:
            os.makedirs('data', exist_ok=True)
            reviews_file = os.path.join('data', f'{app_id.replace(".", "_")}_reviews.jsonl')
            with open(reviews_file, 'w', encoding='utf-8') as f:
                for review in result:
                    f.write(json.dumps(review) + '\n')
        except Exception as save_error:
            print(f"Error saving reviews: {save_error}")
        
        return result
    except Exception as e:
        print(f"Error scraping reviews: {e}")
        # Try to load from local data if scraping fails
        try:
            reviews_file = os.path.join('data', f'{app_id.replace(".", "_")}_reviews.jsonl')
            reviews_data = []
            if os.path.exists(reviews_file):
                with open(reviews_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        reviews_data.append(json.loads(line))
                return reviews_data
            elif os.path.exists('data/sample_reviews.jsonl'):
                with open('data/sample_reviews.jsonl', 'r', encoding='utf-8') as f:
                    for line in f:
                        reviews_data.append(json.loads(line))
                return reviews_data
        except Exception as inner_e:
            print(f"Error loading backup reviews: {inner_e}")
        return []

def categorize_reviews(reviews_list):
    """
    Categorize app reviews into predefined categories using LangChain
    
    Categories:
    - Delivery issue
    - Food stale
    - Delivery partner rude
    - Maps not working properly
    - Instamart should be open all night
    - Bring back 10 minute bolt delivery
    - App issues
    - Price issues
    - Order accuracy issues
    - Good service
    - Other
    
    Args:
        reviews_list: List of review dictionaries
    
    Returns:
        List of reviews with categories added
    """
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0)
    
    # Create a prompt template
    template = """You are a review categorization expert for food delivery apps.
    
    Categorize the following review into exactly ONE of these categories:
    - Delivery issue
    - Food stale
    - Delivery partner rude
    - Maps not working properly
    - Instamart should be open all night
    - Bring back 10 minute bolt delivery
    - App issues
    - Price issues
    - Order accuracy issues
    - Good service
    - Other (if it doesn't fit any of the above)
    
    Review: {review}
    
    Category:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain
    chain = prompt | llm | StrOutputParser()
    
    # Individual processing
    categorized_reviews = []
    
    for review in reviews_list:
        # Extract the content of the review
        review_content = review.get('content', '')
        
        # Skip empty reviews
        if not review_content:
            review_with_category = review.copy()
            review_with_category['category'] = "No content"
            categorized_reviews.append(review_with_category)
            continue
            
        # Categorize the review
        try:
            category = chain.invoke({"review": review_content})
            # Clean up the category response if needed
            category = category.strip()
            
            # Add the category to the review dictionary
            review_with_category = review.copy()
            review_with_category['category'] = category
            categorized_reviews.append(review_with_category)
            
        except Exception as e:
            print(f"Error categorizing review: {e}")
            # Add the original review with an error category
            review_with_category = review.copy()
            review_with_category['category'] = "Error in categorization"
            categorized_reviews.append(review_with_category)
    
    return categorized_reviews