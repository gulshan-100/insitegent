from flask import Flask, render_template, request, jsonify, send_file
from app.categorizer import get_category_counts_from_file
from app.data_loader import get_available_dates, load_reviews_data
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import csv
from io import StringIO

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    """Main page with the reviews interface"""
    # Get available dates
    all_dates = get_available_dates()
    
    # Get date range for July 24 to August 9, 2025
    target_dates = []
    
    # Add dates for July 24-31, 2025
    for i in range(24, 32):  # July 24-31
        date_str = f"2025-07-{i}"
        if date_str in all_dates:
            target_dates.append(date_str)
    
    # Add dates for August 1-9, 2025
    for i in range(1, 10):  # August 1-9
        date_str = f"2025-08-0{i}" if i < 10 else f"2025-08-{i}"
        if date_str in all_dates:
            target_dates.append(date_str)
    
    # If no target dates found, use available dates
    if not target_dates and all_dates:
        target_dates = all_dates[:9]  # Use up to 9 dates
    
    # Get category counts for all dates in the range
    categories = set()
    date_category_counts = {}
    
    for date in target_dates:
        result = load_reviews_data(date)
        if result:
            counts, _ = result
            date_category_counts[date] = counts
            # Collect all unique categories
            categories.update(counts.keys())
    
    # Convert to a sorted list of categories
    categories_list = sorted(list(categories))
    
    return render_template('index.html', 
                          all_dates=all_dates,
                          target_dates=target_dates,
                          categories=categories_list,
                          date_category_counts=date_category_counts)

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

@app.route('/export_to_csv', methods=['GET'])
def export_to_csv():
    """Export the review categories data to a CSV file"""
    # Get available dates
    all_dates = get_available_dates()
    
    # Get date range for July 24 to August 9, 2025
    target_dates = []
    
    # Add dates for July 24-31, 2025
    for i in range(24, 32):  # July 24-31
        date_str = f"2025-07-{i}"
        if date_str in all_dates:
            target_dates.append(date_str)
    
    # Add dates for August 1-9, 2025
    for i in range(1, 10):  # August 1-9
        date_str = f"2025-08-0{i}" if i < 10 else f"2025-08-{i}"
        if date_str in all_dates:
            target_dates.append(date_str)
    
    # If no target dates found, use available dates
    if not target_dates and all_dates:
        target_dates = all_dates[:9]  # Use up to 9 dates
    
    # Get category counts for all dates in the range
    categories = set()
    date_category_counts = {}
    
    for date in target_dates:
        result = load_reviews_data(date)
        if result:
            counts, _ = result
            date_category_counts[date] = counts
            # Collect all unique categories
            categories.update(counts.keys())
    
    # Convert to a sorted list of categories
    categories_list = sorted(list(categories))
    
    # Create a CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"review_categories_{timestamp}.csv"
    filepath = os.path.join("output", filename)
    
    # Make sure output directory exists
    os.makedirs("output", exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row with date columns
        header = ['Category'] + [f"{date[8:]}/{date[5:7]}" for date in target_dates]  # Format as DD/MM
        writer.writerow(header)
        
        # Write data rows
        for category in categories_list:
            row = [category]
            for date in target_dates:
                count = date_category_counts.get(date, {}).get(category, 0)
                row.append(count)
            writer.writerow(row)
    
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/export_category_reviews', methods=['POST'])
def export_category_reviews():
    """Export reviews for a specific category and date to a CSV file"""
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
    
    # Create a CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{category}_{date}_{timestamp}.csv"
    filepath = os.path.join("output", filename)
    
    # Make sure output directory exists
    os.makedirs("output", exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(['Content', 'Score', 'User', 'Timestamp'])
        
        # Write review data
        for review in categorized_reviews[category]:
            writer.writerow([
                review.get('content', ''),
                review.get('score', ''),
                review.get('userName', 'Anonymous'),
                review.get('at', '')
            ])
    
    return jsonify({
        'status': 'success',
        'message': 'CSV file created successfully',
        'filename': filename,
        'download_url': f'/download_file/{filename}'
    })

@app.route('/download_file/<filename>')
def download_file(filename):
    """Download a specific file from the output directory"""
    filepath = os.path.join("output", filename)
    if not os.path.exists(filepath):
        return jsonify({
            'status': 'error',
            'message': 'File not found'
        }), 404
    
    return send_file(filepath, as_attachment=True, download_name=filename)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True)



