import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from preprocess import clean_text
import os

# Configuration
DATA_URL = "https://raw.githubusercontent.com/julialaskar/Customer-Support-Ticket-Analysis/refs/heads/main/customer_support_tickets.csv"
CATEGORY_MODEL_PATH = "category_model.joblib"
PRIORITY_MODEL_PATH = "priority_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

def evaluate():
    print("Loading models and vectorizer...")
    cat_model = joblib.load(CATEGORY_MODEL_PATH)
    prio_model = joblib.load(PRIORITY_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    print("Loading test data...")
    df = pd.read_csv(DATA_URL)
    
    # Preprocessing
    print("Preprocessing text...")
    df['cleaned_description'] = df['Ticket Description'].apply(clean_text)
    
    X = df['cleaned_description']
    y_cat = df['Ticket Type']
    y_prio = df['Ticket Priority']
    
    # Use the same split logic
    _, X_test, _, y_cat_test, _, y_prio_test = train_test_split(
        X, y_cat, y_prio, test_size=0.2, random_state=42
    )
    
    X_test_vec = vectorizer.transform(X_test)
    
    # Category Evaluation
    print("Evaluating Category model...")
    cat_preds = cat_model.predict(X_test_vec)
    cm_cat = confusion_matrix(y_cat_test, cat_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cat_model.classes_, yticklabels=cat_model.classes_)
    plt.title('Confusion Matrix: Ticket Category')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_category.png')
    print("Saved confusion_matrix_category.png")
    
    # Priority Evaluation
    print("Evaluating Priority model...")
    prio_preds = prio_model.predict(X_test_vec)
    cm_prio = confusion_matrix(y_prio_test, prio_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_prio, annot=True, fmt='d', cmap='Greens', 
                xticklabels=prio_model.classes_, yticklabels=prio_model.classes_)
    plt.title('Confusion Matrix: Ticket Priority')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_priority.png')
    print("Saved confusion_matrix_priority.png")
    
    # Print reports
    with open('evaluation_report.txt', 'w') as f:
        f.write("=== Support Ticket Classification Evaluation Report ===\n\n")
        f.write("Category Classification Report:\n")
        f.write(classification_report(y_cat_test, cat_preds))
        f.write("\n\nPriority Classification Report:\n")
        f.write(classification_report(y_prio_test, prio_preds))
    
    print("Saved evaluation_report.txt")

if __name__ == "__main__":
    evaluate()
