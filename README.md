# InsiteGent - Swiggy Reviews Analyzer

An AI-powered web application that categorizes and analyzes Swiggy app reviews using vector embeddings and semantic search with LLM fallback.

## Features

- Process reviews from Swiggy CSV data files
- Automatically categorize reviews using semantic vector search with FAISS
- Use OpenAI embeddings for advanced semantic matching
- Dynamically create new categories using LLM when needed
- Display category summary with counts in a simple web interface
- View detailed reviews by clicking on category counts
- Provide feedback on categorization accuracy to improve the system
- Fallback to pattern matching if vector search fails
- Store reviews in date-wise CSV files for historical analysis

## Setup

1. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser to `http://localhost:5000`

## Usage

### Review Analysis
1. Navigate to the home page
2. View the analysis for the July 24 - August 9, 2025 date range
3. The app will display a summary of categories with counts
4. **Click on any count** to view the specific reviews in that category
5. Categories are color-coded (positive in green, negative in red, neutral in yellow)
6. **Click the Export button** to download the complete table as CSV
7. **Click on export buttons in review modals** to download specific category reviews
8. CSV files are stored in the `output/` folder for further analysis

## Categorization Process

The app uses a three-level categorization approach:

1. **Vector-based Semantic Matching (Primary)**:
   - Converts review texts to embeddings using OpenAI
   - Uses FAISS vector store for efficient similarity search
   - Matches reviews to the most semantically similar predefined category

2. **LLM-based Categorization (Secondary)**:
   - For reviews that don't match any existing category well
   - Uses OpenAI to suggest new categories based on content
   - **Dynamically creates and persists new categories** as needed
   - Consolidates similar topics to avoid redundancy

3. **Pattern-based Matching (Fallback)**:
   - Uses regular expressions for keyword matching
   - Works without API calls if vector search fails
   - Ensures the system always produces results

## Dynamic Category Creation

The system intelligently identifies when new categories are needed:

- Starts with seed categories but evolves based on review content
- Uses GPT models to identify emerging themes and topics
- Persists new categories in `data/dynamic_categories.json`
- Consolidates similar topics to avoid redundant categories (e.g., "Delivery partner rude" and "Delivery person impolite")
- Applies newly created categories to future reviews

## Predefined Categories

The system starts with these seed categories:
- Delivery issue
- Food stale
- Delivery partner rude
- Maps not working properly
- Instamart should be open all night
- Bring back 10 minute bolt delivery
- App issues
- High Charges/Fees
- Payment issues
- Positive Feedback

## Technical Architecture

The project is organized into modular components:
- `app/embedding_utils.py`: Handles OpenAI embeddings for semantic understanding
- `app/vector_store.py`: Implements FAISS vector store for efficient similarity search
- `app/llm_categorizer.py`: Handles LLM-based categorization for reviews that don't match existing categories
- `app/dynamic_category_manager.py`: Manages the creation and persistence of new categories
- `app/categorizer.py`: Main categorization logic integrating all components
- `app/data_loader.py`: Loads and manages review data from CSV files
- `app.py`: Flask web application with REST API endpoints
- `templates/index.html`: Interactive UI with review visualization and feedback mechanisms

### Categorization Process Flow:
1. Load reviews from CSV files
2. Generate embeddings using OpenAI API
3. Compare with predefined and dynamic categories using FAISS similarity search
4. For reviews that don't match well (below threshold), use LLM to suggest categories
5. Create new categories when needed and save them for future use
6. Provide results with interactive UI for exploration and feedback
7. Collect user feedback to continuously improve categorization

## Dependencies

- Flask for web interface
- FAISS for vector search
- OpenAI for embeddings and LLM categorization
- Pandas for data processing
- Python-dotenv for environment management
