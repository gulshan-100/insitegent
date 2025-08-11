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
2. Select a date from the dropdown menu
3. The app will display a summary of categories with counts
4. Categories are color-coded (positive in green, negative in red, neutral in yellow)
5. CSV files are stored in the `swiggy_reviews/` folder

## Categorization Process

The app uses a three-level categorization approach:

1. **Vector-based Semantic Matching (Primary)**:
   - Converts review texts to embeddings using OpenAI
   - Uses FAISS vector store for efficient similarity search
   - Matches reviews to the most semantically similar predefined category

2. **LLM-based Categorization (Secondary)**:
   - For reviews that don't match any existing category well
   - Uses OpenAI to suggest new categories based on content
   - Dynamically expands the category list

3. **Pattern-based Matching (Fallback)**:
   - Uses regular expressions for keyword matching
   - Works without API calls if vector search fails
   - Ensures the system always produces results

## Predefined Categories

The system starts with these predefined categories:
- Delivery issue
- Food stale
- Delivery partner rude
- Maps not working properly
- Instamart should be open all night
- Bring back 10 minute bolt delivery
- App issues
- High Charges/Fees
- Positive Feedback
- Other

## Technical Architecture

The project is organized into modular components:
- `app/embedding_utils.py`: Handles OpenAI embeddings for semantic understanding
- `app/vector_store.py`: Implements FAISS vector store for efficient similarity search
- `app/llm_categorizer.py`: Handles LLM-based categorization for reviews that don't match existing categories
- `app/categorizer.py`: Main categorization logic integrating all components
- `app/data_loader.py`: Loads and manages review data from CSV files
- `app.py`: Flask web application with REST API endpoints
- `templates/index.html`: Interactive UI with review visualization and feedback mechanisms

### Categorization Process Flow:
1. Load reviews from CSV files
2. Generate embeddings using OpenAI API
3. Compare with predefined category examples using FAISS similarity search
4. For reviews that don't match well (below threshold), use LLM to suggest categories
5. Provide results with interactive UI for exploration and feedback
6. Collect user feedback to continuously improve categorization

## Dependencies

- Flask for web interface
- FAISS for vector search
- OpenAI for embeddings and LLM categorization
- Pandas for data processing
- Python-dotenv for environment management
