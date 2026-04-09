import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from preprocess import clean_text
import os

# Configuration
DATA_URL = "https://raw.githubusercontent.com/julialaskar/Customer-Support-Ticket-Analysis/refs/heads/main/customer_support_tickets.csv"
CATEGORY_MODEL_PATH = "category_model.joblib"
PRIORITY_MODEL_PATH = "priority_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

def train():
    print("Downloading dataset...")
    df = pd.read_csv(DATA_URL)
    
    print(f"Dataset loaded: {len(df)} rows.")
    
    # Preprocessing
    print("Preprocessing text...")
    df['cleaned_description'] = df['Ticket Description'].apply(clean_text)
    
    # We will use 'cleaned_description' as the feature
    X = df['cleaned_description']
    y_cat = df['Ticket Type']
    y_prio = df['Ticket Priority']
    
    # Split the data
    X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
        X, y_cat, y_prio, test_size=0.2, random_state=42
    )
    
    # Initialize and fit TF-IDF Vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Category Model
    print("Training Category Classification model...")
    cat_model = LinearSVC(random_state=42)
    cat_model.fit(X_train_vec, y_cat_train)
    
    # Train Priority Model
    print("Training Priority Classification model...")
    prio_model = LinearSVC(random_state=42)
    prio_model.fit(X_train_vec, y_prio_train)
    
    # Basic Evaluation
    print("\nCategory Classification Report:")
    cat_preds = cat_model.predict(X_test_vec)
    print(classification_report(y_cat_test, cat_preds))
    
    print("\nPriority Classification Report:")
    prio_preds = prio_model.predict(X_test_vec)
    print(classification_report(y_prio_test, prio_preds))
    
    # Save Artifacts
    print("Saving models and vectorizer...")
    joblib.dump(cat_model, CATEGORY_MODEL_PATH)
    joblib.dump(prio_model, PRIORITY_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print("Training complete!")

if __name__ == "__main__":
    train()
