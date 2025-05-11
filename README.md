# Fake News Detection using NLP and Machine Learning

This repository contains the complete code, models, and documentation for a final year university project aimed at detecting fake news headlines using a combination of traditional machine learning and deep learning techniques.

## About the Project

The spread of misinformation has become a major threat in today's digital age. This project presents an intelligent system capable of classifying news headlines as **real** or **fake** using natural language processing (NLP) techniques and machine learning models.

The project uses a custom dataset scraped from online news sources and applies both traditional ML algorithms (like Logistic Regression and Naive Bayes) and deep learning models (CNN, RNN) for accurate classification.


## Features

- Text Preprocessing (cleaning, tokenization, stopwords removal)
- Feature Extraction using **TF-IDF** and **Bag of Words**
- Multiple ML Models: Logistic Regression, Naive Bayes
- Deep Learning Models: CNN and RNN using Keras
- Evaluation using accuracy, precision, recall, F1-score
- Visualizations: Confusion Matrix, Accuracy Comparison Charts
- Modular pipeline for easy experimentation and improvements


## Tech Stack

- Python 3.x
- scikit-learn
- Keras / TensorFlow
- pandas / NumPy
- matplotlib / seaborn
- Jupyter Notebook


## 🗃 Dataset

- News headlines scraped from public sources
- Balanced and cleaned to ensure reliable training
- Labeled as `true` or `false`

## 📁 Folder Structure

project-root/
├── data/ # Preprocessed dataset
├── models/ # Trained ML and DL models (optional)
├── notebooks/ # Jupyter notebooks for experiments
├── results/ # Screenshots and evaluation outputs
├── diagrams/ # System design and DFD diagrams
├── README.md # This file
└── Final_Report.docx # Full dissertation report

## 🚀 Getting Started

### 1. Clone the repository

git clone https://github.com/yourusername/fake-news-detection-nlp.git
cd fake-news-detection-nlp
2. Install dependencies
pip install -r requirements.txt
Or manually install:

pip install pandas numpy scikit-learn keras matplotlib seaborn nltk
3. Run the Notebook
Open the main Jupyter notebook and run all cells:

jupyter notebook
Results Summary
Model	Accuracy	F1 Score
Logistic Regression TF	91.91%	89.86%
Naive Bayes BoW	90.63%	88.50%
CNN	90.64%	89.40%
RNN	89.79%	88.10%

System Architecture
Includes system design and data flow diagrams in diagrams/.

 DFD Level 0 and Level 1

 System Design (modular pipeline)

 Preprocessing → Feature Extraction → Model → Output

Future Work
Extend to full article detection (not just headlines)

Integrate BERT / Transformer models

Add explainability (XAI) with SHAP or LIME

Build web or mobile app for real-time fact-checking

Support for multiple languages

