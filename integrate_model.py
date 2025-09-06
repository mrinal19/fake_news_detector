"""
Integration script to connect your existing fake_news_detector.py with the Flask backend
Run this script to train/load your model and prepare it for the web interface
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re
import string
import os

def preprocess_text(text):
    """Preprocess text for better model performance"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#','', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_train_model(dataset_path=None):
    """Load dataset and train the model"""
    print("🔄 Starting model training process...")
    
    # Try to load your dataset
    if dataset_path and os.path.exists(dataset_path):
        print(f"📊 Loading dataset from {dataset_path}")
        try:
            # Try different common formats
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            else:
                print("❌ Unsupported file format. Please use CSV or JSON.")
                return False
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    else:
        print("📊 No dataset provided. Creating sample dataset for demonstration...")
        # Create a larger sample dataset
        fake_samples = [
            "BREAKING: Government secretly controlling weather with hidden technology!",
            "Doctors HATE this one weird trick for instant weight loss!",
            "Shocking truth about vaccines that pharmaceutical companies don't want you to know!",
            "Local mom discovers miracle cure that works in 24 hours!",
            "Anonymous insider reveals government conspiracy about moon landing!",
            "You won't believe what happens when you mix these two common ingredients!",
            "Secret society of billionaires planning global takeover next month!",
            "Ancient remedy discovered that cures all diseases instantly!",
            "Government officials caught hiding evidence of alien contact!",
            "Miracle superfood that burns fat while you sleep - doctors furious!",
            "Shocking revelation: Celebrity death was faked for publicity!",
            "One simple trick to make millions from home - banks hate it!",
            "Government planning to replace all money with microchips!",
            "Secret ingredient in tap water is making people sick!",
            "Amazing discovery: Time travel finally proven possible!"
        ]
        
        real_samples = [
            "Scientists at Stanford University published breakthrough research on quantum computing in Nature journal.",
            "Federal Reserve announces interest rate adjustment following comprehensive economic analysis.",
            "New clinical trial results show promising outcomes for Alzheimer's treatment, researchers report.",
            "Climate scientists present updated global warming projections at international conference.",
            "Technology sector reports steady growth in quarterly earnings across major companies.",
            "Medical researchers at Johns Hopkins develop improved surgical technique for heart procedures.",
            "Economic analysts forecast moderate growth based on employment and inflation data.",
            "University consortium receives federal grant for renewable energy research initiative.",
            "Peer-reviewed study demonstrates effectiveness of new cancer immunotherapy treatment.",
            "International health organization updates vaccination recommendations based on latest evidence.",
            "Research team publishes findings on biodiversity conservation in tropical ecosystems.",
            "Central bank policy makers discuss monetary strategy in response to economic indicators.",
            "Clinical researchers report successful trial of novel diabetes management approach.",
            "Academic institutions collaborate on artificial intelligence ethics research project.",
            "Government agencies release comprehensive report on infrastructure investment needs."
        ]
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': fake_samples + real_samples,
            'label': [1] * len(fake_samples) + [0] * len(real_samples)  # 1 for fake, 0 for real
        })
    
    # Ensure proper column names
    text_column = None
    label_column = None
    
    # Common column name variations
    text_variations = ['text', 'content', 'article', 'news', 'title', 'headline']
    label_variations = ['label', 'target', 'class', 'fake', 'is_fake']
    
    for col in df.columns:
        if col.lower() in text_variations and text_column is None:
            text_column = col
        if col.lower() in label_variations and label_column is None:
            label_column = col
    
    if not text_column or not label_column:
        print("❌ Could not identify text and label columns. Please ensure your dataset has appropriate column names.")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    print(f"📝 Using '{text_column}' as text column and '{label_column}' as label column")
    print(f"📊 Dataset shape: {df.shape}")
    
    # Preprocess text
    print("🔄 Preprocessing text data...")
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0]
    
    print(f"📊 After preprocessing: {df.shape}")
    print(f"📈 Class distribution:\n{df[label_column].value_counts()}")
    
    # Split data
    X = df['processed_text']
    y = df[label_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create TF-IDF vectorizer
    print("🔄 Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Logistic Regression model
    print("🔄 Training Logistic Regression model...")
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Save model and vectorizer
    print("💾 Saving model and vectorizer...")
    with open('fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save model metadata
    metadata = {
        'accuracy': accuracy,
        'model_type': 'LogisticRegression',
        'vectorizer_type': 'TfidfVectorizer',
        'max_features': vectorizer.max_features,
        'ngram_range': vectorizer.ngram_range,
        'training_samples': len(X_train)
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("✅ Model and vectorizer saved successfully!")
    print("🚀 You can now run the Flask backend with: python app.py")
    
    return True

def test_model():
    """Test the saved model with sample inputs"""
    print("\n🧪 Testing saved model...")
    
    try:
        # Load model and vectorizer
        with open('fake_news_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Test samples
        test_samples = [
            "Scientists at MIT published groundbreaking research in peer-reviewed journal Nature.",
            "SHOCKING: Government hiding alien technology from public for decades!"
        ]
        
        for i, text in enumerate(test_samples):
            processed_text = preprocess_text(text)
            text_vector = vectorizer.transform([processed_text])
            prediction = model.predict(text_vector)[0]
            probability = model.predict_proba(text_vector)[0]
            confidence = max(probability) * 100
            
            result = "FAKE" if prediction == 1 else "REAL"
            
            print(f"\n📰 Test {i+1}: {text[:50]}...")
            print(f"🔍 Prediction: {result}")
            print(f"📊 Confidence: {confidence:.1f}%")
            print(f"📈 Probabilities: Real={probability[0]*100:.1f}%, Fake={probability[1]*100:.1f}%")
        
        print("✅ Model testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Fake News Detector - Model Integration")
    print("=" * 50)
    
    # Ask user for dataset path
    dataset_path = input("\n📂 Enter path to your dataset (or press Enter to use sample data): ").strip()
    
    if dataset_path and not os.path.exists(dataset_path):
        print(f"❌ File not found: {dataset_path}")
        print("🔄 Using sample data instead...")
        dataset_path = None
    
    # Train the model
    success = load_and_train_model(dataset_path)
    
    if success:
        # Test the model
        test_model()
        
        print("\n" + "=" * 50)
        print("🎉 Setup Complete! Next steps:")
        print("1. Save the HTML UI as 'index.html' in the same directory")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Run the backend: python app.py")
        print("4. Open http://localhost:5000 in your browser")
        print("=" * 50)
    else:
        print("❌ Model training failed. Please check your dataset and try again.")