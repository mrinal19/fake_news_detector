from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string
import time
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global variables for model and vectorizer
model = None
vectorizer = None
model_accuracy = 0.92  # Default accuracy, update this with your actual model accuracy

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer, model_accuracy
    
    try:
        # Try to load existing model and vectorizer
        if os.path.exists('fake_news_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
            with open('fake_news_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            print("✅ Model and vectorizer loaded successfully!")
            return True
        else:
            print("❌ Model files not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def preprocess_text(text):
    """Preprocess text for prediction"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove numbers (optional, depending on your training)
    text = re.sub(r'\d+', '', text)
    
    return text

def train_model_if_needed():
    """Train a basic model if no model exists (for demo purposes)"""
    global model, vectorizer, model_accuracy
    
    print("🔄 Training a basic model for demonstration...")
    
    # Sample training data (replace with your actual dataset)
    fake_news_samples = [
        "BREAKING: Shocking secret that doctors don't want you to know!",
        "Miracle cure discovered! Government trying to hide this information!",
        "You won't believe this weird trick that works instantly!",
        "Anonymous sources reveal shocking conspiracy about vaccines!",
        "Local mom discovers one simple trick to lose weight fast!",
        "Scientists hate him! Man discovers secret to eternal youth!",
        "Government officials planning to control your mind with 5G!",
        "This ancient remedy cures everything! Doctors are furious!",
        "Shocking revelation: Moon landing was completely fake!",
        "Celebrity death hoax spreads across social media rapidly!"
    ]
    
    real_news_samples = [
        "Scientists at MIT published research on quantum computing advances in Nature journal.",
        "The Federal Reserve announced interest rate changes following economic data analysis.",
        "New study from Harvard Medical School shows promising results for cancer treatment.",
        "Climate researchers present findings at international conference on global warming.",
        "Technology companies report quarterly earnings showing steady growth patterns.",
        "University researchers collaborate on breakthrough in renewable energy storage.",
        "Medical professionals recommend updated vaccination schedules based on latest evidence.",
        "Economic analysts predict market trends based on comprehensive data analysis.",
        "Research institutions receive federal funding for advanced scientific studies.",
        "Peer-reviewed study demonstrates effectiveness of new medical treatment protocol."
    ]
    
    # Create training data
    texts = fake_news_samples + real_news_samples
    labels = [1] * len(fake_news_samples) + [0] * len(real_news_samples)  # 1 for fake, 0 for real
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed_texts)
    
    # Train Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X, labels)
    
    # Save the model and vectorizer
    with open('fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("✅ Basic model trained and saved!")
    return True

@app.route('/')
def home():
    """Serve the main HTML page"""
    # Read the HTML file content (you'll need to save the UI as index.html)
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return """
        <h1>Fake News Detector API</h1>
        <p>Backend is running! Please place your index.html file in the same directory.</p>
        <p>API Endpoint: POST /predict</p>
        <p>Model Status: {'Loaded' if model else 'Not Loaded'}</p>
        """

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if news is fake or real"""
    start_time = time.time()
    
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        news_text = data['text'].strip()
        
        if len(news_text) < 10:
            return jsonify({'error': 'Text too short for analysis'}), 400
        
        if not model or not vectorizer:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Preprocess the text
        processed_text = preprocess_text(news_text)
        
        # Vectorize the text
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        prediction_proba = model.predict_proba(text_vector)[0]
        
        # Get confidence score
        confidence = max(prediction_proba) * 100
        
        # Calculate additional metrics
        word_count = len(news_text.split())
        processing_time = round(time.time() - start_time, 3)
        
        # Prepare response
        result = {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': round(confidence, 1),
            'probability_fake': round(prediction_proba[1] * 100, 1),
            'probability_real': round(prediction_proba[0] * 100, 1),
            'word_count': word_count,
            'processing_time': processing_time,
            'model_accuracy': round(model_accuracy * 100, 1)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'model_accuracy': round(model_accuracy * 100, 1)
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    return jsonify({
        'model_accuracy': round(model_accuracy * 100, 1),
        'model_type': 'Logistic Regression',
        'vectorizer_type': 'TF-IDF',
        'max_features': getattr(vectorizer, 'max_features', 'N/A') if vectorizer else 'N/A'
    })

if __name__ == '__main__':
    print("🚀 Starting Fake News Detector Backend...")
    
    # Try to load existing model first
    if not load_model():
        print("🔄 No existing model found. Training basic model...")
        train_model_if_needed()
    
    print("🌐 Server starting on http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/api/predict")
    print("❤️  Health check: http://localhost:5000/api/health")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)