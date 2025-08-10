#use the scrapper to scrape the reviews of the app and show it in the interface in html css file ijdex.html

from app.scrapper import scrape_reviews
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    """Main page with the reviews interface"""
    return render_template('index.html')

@app.route('/fetch_reviews')
def fetch_reviews():
    """API endpoint to fetch reviews for a specific app"""
    app_id = request.args.get('app_id')
    count = int(request.args.get('count', 10))
    
    if not app_id:
        return jsonify({
            'status': 'error',
            'message': 'Missing app_id parameter'
        }), 400
    
    if count < 1 or count > 200:
        return jsonify({
            'status': 'error',
            'message': 'Count must be between 1 and 200'
        }), 400
    
    try:
        # Scrape reviews using the scrapper
        reviews = scrape_reviews(app_id, count)
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully fetched {len(reviews)} reviews',
            'app_id': app_id,
            'reviews': reviews
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error fetching reviews: {str(e)}',
            'app_id': app_id
        }), 500

if __name__ == '__main__':
    app.run(debug=True)




