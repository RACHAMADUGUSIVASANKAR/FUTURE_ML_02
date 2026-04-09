# Support AI: Ticket Classification and Prioritization

An intelligent Machine Learning system that automates help-desk ticket categorization and prioritization, enabling faster response times and efficient support workflows.

---

## Project Overview

Support teams often receive large volumes of unstructured customer tickets. Manually reviewing and routing them slows response time and increases operational overhead.

Support AI addresses this problem by automatically:

* Categorizing tickets into issue types such as Technical Issue, Billing, and Account Access
* Assigning urgency levels such as High, Medium, and Low
* Improving response efficiency using Natural Language Processing techniques

This project demonstrates a complete machine learning pipeline including preprocessing, training, evaluation, and deployment through an interactive dashboard.

---

## Key Features

* Automatic ticket classification
* Priority prediction based on ticket content
* NLP preprocessing using NLTK
* TF-IDF feature extraction
* LinearSVC classification models
* Performance evaluation with confusion matrices
* Interactive Streamlit web interface
* Persistent trained models using Joblib

---

## Technology Stack

| Tool                   | Purpose                                 |
| ---------------------- | --------------------------------------- |
| Python 3.13            | Core programming language               |
| Scikit-Learn           | Model training and TF-IDF vectorization |
| NLTK                   | Text preprocessing                      |
| Pandas                 | Data manipulation                       |
| Streamlit              | Web interface                           |
| Joblib                 | Model persistence                       |
| Matplotlib and Seaborn | Visualization                           |

---

## Project Structure

```
Support-AI/
│
├── preprocess.py        # Text cleaning pipeline
├── train_model.py       # Model training and saving
├── evaluate.py          # Performance evaluation scripts
├── app.py               # Streamlit dashboard
├── models/              # Saved trained models
├── data/                # Dataset storage (if applicable)
└── README.md            # Project documentation
```

---

## Installation Guide

### Step 1: Clone the Repository

```
git clone https://github.com/yourusername/support-ai-ticket-classifier.git
cd support-ai-ticket-classifier
```

### Step 2: Install Dependencies

```
pip install pandas scikit-learn nltk streamlit joblib matplotlib seaborn
```

---

## Model Training

Run the following command to train the models:

```
python train_model.py
```

This step performs dataset loading, preprocessing, model training, and model saving.

---

## Model Evaluation

Run the evaluation script:

```
python evaluate.py
```

This generates:

* Precision scores
* Recall scores
* F1-scores
* Confusion matrix visualizations

---

## Run the Web Application

Start the Streamlit application:

```
streamlit run app.py
```

Then open the local URL displayed in the terminal (typically http://localhost:8501).

The interface allows users to enter ticket text and receive predicted category and priority level.

---

## Example Use Case

Input:

"I cannot access my account after resetting my password."

Output:

Category: Account Access
Priority: High

---

## Future Improvements

* Integration of transformer-based models such as BERT
* Support for multilingual ticket classification
* REST API deployment using FastAPI
* Cloud deployment support
* Real-time ticket routing integration

---

## Author

Developed for Future Interns – Machine Learning Task 2 (2026)

Project Code: FUTURE_ML_02

---

## License

This project is intended for educational and internship evaluation purposes.
