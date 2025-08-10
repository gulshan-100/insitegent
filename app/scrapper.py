from google_play_scraper import reviews, Sort

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
        
        return result
    except Exception as e:
        print(f"Error scraping reviews: {e}")
        return []