import streamlit as st
import joblib
import pandas as pd
from preprocess import clean_text
import os

# Page configuration
st.set_page_config(
    page_title="Support AI: Ticket Classifier & Prioritizer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .priority-high { color: #dc3545; font-weight: bold; }
    .priority-medium { color: #ffc107; font-weight: bold; }
    .priority-low { color: #28a745; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Helper function to load models
@st.cache_resource
def load_artifacts():
    cat_model = joblib.load("category_model.joblib")
    prio_model = joblib.load("priority_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return cat_model, prio_model, vectorizer

def main():
    st.title("🔍 Support AI")
    st.subheader("Intelligent Support Ticket Classification & Prioritization")
    
    # Check if models exist
    if not os.path.exists("category_model.joblib"):
        st.error("Model files not found. Please run `train_model.py` first.")
        return

    cat_model, prio_model, vectorizer = load_artifacts()

    # Sidebar
    st.sidebar.header("About the System")
    st.sidebar.info("""
    This system uses Machine Learning to:
    1. **Classify** support tickets into categories.
    2. **Prioritize** issues (High/Medium/Low).
    
    Built with Python, Scikit-Learn, and Streamlit.
    """)
    
    if os.path.exists("evaluation_report.txt"):
        st.sidebar.markdown("### Model Performance")
        with open("evaluation_report.txt", "r") as f:
            report = f.read()
            st.sidebar.text_area("Metrics Summary", report, height=300)

    # Main area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        ticket_text = st.text_area("Enter Support Ticket Description:", 
                                   placeholder="e.g., I'm having trouble logging into my account. The password reset link is not working.", 
                                   height=200)
        
        if st.button("Analyze Ticket"):
            if ticket_text.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                # Preprocess
                cleaned = clean_text(ticket_text)
                vec = vectorizer.transform([cleaned])
                
                # Predict
                category = cat_model.predict(vec)[0]
                priority = prio_model.predict(vec)[0]
                
                st.markdown("---")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Predicted Category", category)
                
                with res_col2:
                    p_class = f"priority-{priority.lower()}"
                    st.markdown(f"### Predicted Priority: <span class='{p_class}'>{priority}</span>", unsafe_allow_html=True)
                
                st.success("Analysis complete! Ticket routed successfully.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### System Insights")
        st.write("Current Categories Supported:")
        st.code(", ".join(cat_model.classes_))
        
        st.write("Priority Levels:")
        st.code(", ".join(prio_model.classes_))
        
        if os.path.exists("confusion_matrix_category.png"):
             st.image("confusion_matrix_category.png", caption="Category Performance", use_container_width=True)

if __name__ == "__main__":
    main()
