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
        category_counts = load_reviews_data(selected_date)
    
    return render_template('index.html', 
                          dates=dates, 
                          selected_date=selected_date,
                          category_counts=category_counts)

@app.route('/api/reviews/<date>')
def get_reviews(date):
    """API endpoint to get reviews for a specific date"""
    category_counts = load_reviews_data(date)
    
    if category_counts:
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

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True)



