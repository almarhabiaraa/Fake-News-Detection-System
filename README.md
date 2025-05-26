# Natural Language Processing (NLP) Course

Welcome to the **Natural Language Processing (NLP)** course repository! This repository contains a series of hands-on labs and a final project that explore core NLP concepts, techniques, and real-world applications using Python, Pandas, Scikit-learn, and Deep Learning.

---

## Repository Structure

```bash
.
├── labs/         # Hands-on NLP lab exercises
│   ├── lab1/     # Data Analysis with Pandas
│   ├── lab2/     # NLP Pipeline + Text Preprocessing + BoW & TF-IDF
│   ├── lab3/     # Regex & Word Embedding
│   ├── lab4/     # Data-Centric vs Model-Centric + Product Review Classifier
│   ├── lab5/     # Topic Modeling Using LDA
│   └── lab6/     # Generative AI Use Case: Dialogue Summarization
│
└── project/      # Final project: Fake News Detection
    ├── notebook/         # Jupyter notebooks for data analysis and modeling
    ├── project_report/   # Written report explaining methods and findings
    └── presentation/     # Final presentation slides

```

---

## Labs Overview

Each lab focuses on a different aspect of NLP:

### Lab 1 – Data Analysis with Pandas
- Learn the foundations of data analysis using the **Pandas** library, tailored for working with text datasets.

### Lab 2 – Building an NLP Pipeline
- **Dataset**: Twitter  
- **Tasks**:
  - Raw text preprocessing
  - Cleaning and normalization
  - Bag of Words (BoW) and TF-IDF feature extraction

### Lab 3 – Regex and Word Embedding
- Use **Regular Expressions** for text pattern extraction.
- Apply **Word2Vec** to capture semantic meaning through word embeddings.

### Lab 4 – Data-Centric vs Model-Centric Approaches
- **Focus**: Product review classification (magazine category)
- Understand the difference between:
  - **Data-centric** approaches: improving data quality
  - **Model-centric** approaches: enhancing model performance

### Lab 5 – Topic Modeling Using LDA
- Learn how to uncover hidden topics in a text corpus using **Latent Dirichlet Allocation (LDA)**.

### Lab 6 – Generative AI Use Case
- **Task**: Summarize dialogues
- Leverage **generative models** to understand and compress conversational content.

---

## Final Project – Fake News Detection

This project explores the application of machine learning and deep learning techniques for detecting fake news.

### Dataset
- **Source**: Kaggle Fake News Dataset (labeled real vs fake articles)

### Techniques Used
- **Text Representation**: TF-IDF, Word2Vec  
- **Classification Models**: Logistic Regression, XGBoost  
- **Advanced Modeling**: BERT-based topic modeling  
- **Evaluation Metrics**: Accuracy, F1-score, and other performance indicators

## Model Performance Comparison

### Traditional Machine Learning Models

| Model               | Accuracy (%) |
|---------------------|--------------|
| Logistic Regression | 98.74        |
| Decision Tree       | 99.51        |
| SVM                 | 99.36        |
| Gradient Boosting   | 99.41        |
| XGBoost             | 99.69        |
| Random Forest       | 99.57        |
| Naive Bayes         | 93.06        |

### Deep Learning Models

| Model | Accuracy (%) | Notes |
|-------|--------------|-------|
| LSTM  | 97.59        | Strong at capturing sequential dependencies |
| GRU   | 94.99        | Slightly lower than LSTM but still effective |
| CNN   | 98.95        | Highest accuracy; great at local pattern detection |


### Key Findings
- **XGBoost** and **CNN** models achieved high accuracy and strong generalization.
- Combining traditional ML with DL techniques improves detection performance significantly.


### Project Contents

- `notebook/` – Complete codebase and exploratory analysis  
- `project_report/` – Detailed explanation of methods, models, and findings  
- `presentation/` – Visual slides summarizing project goals, workflow, and outcomes

---
## Skills Gained
- Text preprocessing & feature extraction
- Sentiment and topic classification
- Word embeddings & semantic representation
- Regular expressions for NLP
- Topic modeling with LDA
- Generative AI and summarization
- Fake news detection with classical and deep models

--- 
## Author
Developed by **Araa Almarhabi** as part of the **Natural Language Processing** course.







