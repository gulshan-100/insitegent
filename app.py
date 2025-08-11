from flask import Flask, render_template, request, jsonify
from app.categorizer import get_category_counts_from_file
from app.data_loader import get_available_dates, load_reviews_data
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    """Main page with the reviews interface"""
    # Get available dates
    dates = get_available_dates()
    
    # Default to the first date if available
    selected_date = request.args.get('date')
    if not selected_date and dates:
        selected_date = dates[0]
    
    # Get category counts for the selected date
    category_counts = {}
    if selected_date:
        result = load_reviews_data(selected_date)
        if result:
            category_counts, _ = result
    
    return render_template('index.html', 
                          dates=dates, 
                          selected_date=selected_date,
                          category_counts=category_counts)

@app.route('/api/reviews/<date>')
def get_reviews(date):
    """API endpoint to get reviews for a specific date"""
    result = load_reviews_data(date)
    
    if result:
        category_counts, _ = result
        return jsonify({
            'status': 'success',
            'date': date,
            'categories': category_counts
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'No data available for {date}'
        }), 404

@app.route('/get_category_reviews', methods=['POST'])
def get_category_reviews():
    """API endpoint to get reviews for a specific category"""
    data = request.get_json()
    date = data.get('date')
    category = data.get('category')
    
    if not date or not category:
        return jsonify({
            'status': 'error',
            'message': 'Missing date or category parameter'
        }), 400
    
    result = load_reviews_data(date)
    if not result:
        return jsonify({
            'status': 'error',
            'message': f'No data available for {date}'
        }), 404
    
    _, categorized_reviews = result
    
    if category not in categorized_reviews:
        return jsonify({
            'status': 'error',
            'message': f'Category {category} not found'
        }), 404
    
    return jsonify({
        'status': 'success',
        'date': date,
        'category': category,
        'reviews': categorized_reviews[category]
    })
    
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """API endpoint to receive feedback on categorization"""
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400
        
    data = request.get_json()
    
    # Required fields
    category = data.get('category')
    review_text = data.get('reviewText')
    is_correct = data.get('isCorrect')
    suggested_category = data.get('suggestedCategory')
    
    if not all([category, review_text]):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    
    # In a real implementation, you would:
    # 1. Store this feedback in a database
    # 2. Use it to improve the categorization model
    # 3. Periodically retrain the system with this feedback
    
    # For now, just log it
    app.logger.info(f"Feedback received: Category='{category}', IsCorrect={is_correct}, Review='{review_text[:30]}...'")
    
    # This could be expanded to:
    # - Add examples to the predefined categories
    # - Adjust similarity thresholds
    # - Create training data for the LLM
    
    return jsonify({'status': 'success', 'message': 'Feedback received'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True)



