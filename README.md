# E-Commerce Product Review Sentiment Analysis

A complete end-to-end web application for analyzing product reviews using AI-powered sentiment analysis with BERT and aspect-based extraction.

## Features

### Core Functionality
- **Single Review Analysis**: Analyze individual reviews with detailed sentiment scores
- **Aspect-Based Extraction**: Automatically identify product strengths and weaknesses
- **Batch Processing**: Analyze multiple reviews simultaneously
- **Analytics Dashboard**: Visualize sentiment distribution with interactive charts
- **Real-time Analysis**: Instant predictions using pre-trained BERT models

### Technical Highlights
- **BERT Model**: DistilBERT fine-tuned on SST-2 for sentiment classification
- **Aspect Detection**: Rule-based extraction of 10+ product aspects
- **Data Persistence**: Supabase integration for storing review analytics
- **Modern UI**: React with Tailwind CSS and Bootstrap components
- **REST API**: Clean Flask backend with JSON endpoints

## Tech Stack

### Backend
- Python 3.8+
- Flask (REST API)
- HuggingFace Transformers (BERT/DistilBERT)
- PyTorch
- Supabase Python Client

### Frontend
- React 18
- TypeScript
- Tailwind CSS
- Chart.js & React-Chartjs-2
- Axios
- Lucide React (icons)

## Project Structure

```
project/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── requirements.txt       # Python dependencies
│   └── README.md             # Backend documentation
├── src/
│   ├── components/
│   │   ├── Header.tsx
│   │   ├── SingleReviewAnalyzer.tsx
│   │   ├── AspectAnalyzer.tsx
│   │   ├── BatchAnalyzer.tsx
│   │   └── Dashboard.tsx
│   ├── services/
│   │   └── api.ts            # API client
│   ├── types.ts              # TypeScript interfaces
│   ├── App.tsx               # Main application
│   └── main.tsx              # Entry point
└── README.md                 # This file
```

## Installation & Setup

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Supabase account (free tier works)

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Download SpaCy model (optional, for enhanced NLP):
```bash
python -m spacy download en_core_web_sm
```

5. Configure environment variables:
Create a `.env` file in the backend directory:
```env
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

6. Start Flask server:
```bash
python app.py
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
Update `.env` file in project root with your Supabase credentials:
```env
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

3. Start development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:5173`

## Usage

### 1. Single Review Analysis
- Navigate to "Single Review" tab
- Enter a product review
- Click "Analyze Sentiment"
- View sentiment classification with confidence scores

### 2. Aspect-Based Analysis
- Navigate to "Aspect Analysis" tab
- Enter a detailed product review
- Click "Extract Aspects"
- View identified strengths and weaknesses with examples

### 3. Batch Processing
- Navigate to "Batch Analysis" tab
- Enter multiple reviews (one per line)
- Click "Analyze All Reviews"
- View results in a sortable table with summary statistics

### 4. Dashboard
- Navigate to "Dashboard" tab
- View overall statistics from all analyzed reviews
- Explore sentiment distribution via pie and bar charts
- Click "Refresh Stats" to update data

## API Documentation

### Endpoints

#### GET `/health`
Check server status and model information.

#### POST `/predict`
Analyze a single review.
```json
Request: { "review": "text" }
Response: { "sentiment": "Positive", "confidence": 95.5 }
```

#### POST `/aspects`
Extract product aspects from review.
```json
Request: { "review": "text" }
Response: {
  "overall_sentiment": {...},
  "strengths": [...],
  "weaknesses": [...]
}
```

#### POST `/batch`
Analyze multiple reviews.
```json
Request: { "reviews": ["review1", "review2"] }
Response: { "results": [...], "summary": {...} }
```

#### GET `/stats`
Get analytics from all stored reviews.
```json
Response: {
  "total": 100,
  "positive": 65,
  "negative": 35,
  "positive_percentage": 65.0
}
```

## Model Information

**Primary Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- Lightweight version of BERT (66M parameters)
- Fine-tuned on Stanford Sentiment Treebank (SST-2)
- Binary classification: Positive/Negative
- ~95% accuracy on test set

**Aspect Detection**: Rule-based keyword matching
- 10 product categories (battery, performance, quality, etc.)
- Sentence-level sentiment for each aspect
- Confidence scores per aspect

## Detected Aspects

The system automatically detects sentiment for:
- Battery life
- Performance/Speed
- Build quality
- Design/Aesthetics
- Price/Value
- Delivery/Shipping
- Screen/Display
- Camera quality
- Sound/Audio
- Customer service

## Database Schema

### reviews table
- `id`: UUID (primary key)
- `review_text`: TEXT (review content)
- `sentiment`: TEXT (Positive/Negative)
- `confidence`: NUMERIC (0-100)
- `created_at`: TIMESTAMP (auto-generated)

## Architecture

### Data Flow
1. User enters review in React frontend
2. Frontend sends POST request to Flask API
3. Flask tokenizes text using BERT tokenizer
4. Model generates sentiment predictions
5. Response sent back to frontend with scores
6. Results stored in Supabase for analytics
7. Dashboard fetches aggregated statistics

### Aspect Extraction Process
1. Review text preprocessed and lowercased
2. Keyword matching against aspect dictionary
3. Extract sentences containing aspect keywords
4. Run sentiment analysis on each sentence
5. Aggregate results by aspect category
6. Return strengths (positive) and weaknesses (negative)

## Building for Production

### Frontend
```bash
npm run build
```
Output will be in `dist/` directory.

### Backend
Use a production WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Troubleshooting

### Backend Issues

**Model Loading Errors**
- Ensure you have stable internet for first-time model download
- Models are cached in `~/.cache/huggingface/`

**Memory Issues**
- DistilBERT requires ~1GB RAM
- Consider using CPU instead of GPU for development

**Port Already in Use**
- Change port in `app.py`: `app.run(port=5001)`

### Frontend Issues

**API Connection Errors**
- Verify Flask server is running on port 5000
- Check CORS is enabled in Flask app
- Ensure no firewall blocking localhost connections

**Chart Not Displaying**
- Clear browser cache
- Check console for Chart.js errors
- Verify data format matches chart requirements

## Performance Optimization

- **Model**: Using DistilBERT (40% smaller, 60% faster than BERT)
- **Caching**: Consider adding Redis for prediction caching
- **Batch Size**: Process up to 100 reviews per batch request
- **Database**: Indexed columns for fast analytics queries

## Future Enhancements

- [ ] Multi-language support
- [ ] Star rating prediction (1-5 stars)
- [ ] Fine-tune model on domain-specific data
- [ ] Export analytics to CSV/PDF
- [ ] Real-time streaming analysis
- [ ] User authentication and saved analyses
- [ ] Comparison between products
- [ ] Sentiment trends over time

## License

MIT License

## Credits

- BERT Model: Google AI
- DistilBERT: HuggingFace
- Frontend Framework: React Team
- UI Icons: Lucide React
