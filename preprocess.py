import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase
    2. Removing special characters and numbers
    3. Removing stopwords
    4. Lemmatization
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenize
    words = text.split()
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    
    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)

if __name__ == "__main__":
    sample_text = "I am having issues with my billing. The invoice #12345 is incorrect!"
    print(f"Original: {sample_text}")
    print(f"Cleaned: {clean_text(sample_text)}")
