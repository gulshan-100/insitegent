import os
import glob
from datetime import datetime
from app.categorizer import get_category_counts_from_file

def get_available_dates(reviews_dir='swiggy_reviews'):
    """Get all available dates from CSV files in the reviews directory"""
    try:
        # Get all CSV files
        csv_files = glob.glob(os.path.join(reviews_dir, '*.csv'))
        
        # Extract dates from filenames
        dates = []
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            date_str = filename.replace('.csv', '')
            try:
                # Parse the date
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date_str)
            except ValueError:
                # Skip files with invalid date format
                continue
        
        # Sort dates
        dates.sort()
        return dates
    except Exception as e:
        print(f"Error getting available dates: {e}")
        return []

def load_reviews_data(date, reviews_dir='swiggy_reviews'):
    """
    Load reviews data for a specific date
    
    Returns:
        Tuple of (category_counts, categorized_reviews) or None if file doesn't exist
    """
    file_path = os.path.join(reviews_dir, f"{date}.csv")
    if not os.path.exists(file_path):
        return None
    
    # Get category counts and categorized reviews
    counts, categorized_reviews = get_category_counts_from_file(file_path)
    return counts, categorized_reviews
