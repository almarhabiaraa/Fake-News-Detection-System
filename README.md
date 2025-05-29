# Fake News Detection System 
**Course:** CS4083 – Text Mining & Natural Language Processing (NLP)  
**Developer:** Araa Almarhabi  


## Introduction
This project investigates the application of machine learning (ML) and deep learning (DL) techniques to automatically detect fake news articles. By leveraging advanced natural language processing (NLP) tools and models, the system aims to classify news content as **real** or **fake** based on textual features.

## Features

- Uses machine learning and deep learning models to detect fake news accurately.
- Converts news articles into numbers using TF-IDF and Word2Vec.
- Organizes the project into clear folders for code, reports, and presentations.
- Compares different models using accuracy and other performance measures.
- Prepares text data by cleaning and processing it before training.
- Supports multiple models like Logistic Regression, XGBoost, CNN, and LSTM.
- Includes BERT-based topic modeling for deeper understanding of articles.
- Shows easy-to-understand charts and graphs to explain results.
- Covers all steps from data loading to final evaluation.

## Repository Structure

```bash
.
└── project/      # Final project: Fake News Detection
    ├── notebook/         # Jupyter notebooks for complete codebase and exploratory analysis  
    ├── project_report/   # Written report explaining methods and findings
    ├── nlp_poster/       # Project Poster 
    └── presentation/     # Final presentation slides
```



### Dataset
- **Source**: Kaggle – Fake News Dataset
- **Content**: Labeled articles classified as real or fake

### Techniques Used
- **Text Representation**: TF-IDF, and Word2Vec  
- **Classification Models**: Logistic Regression, XGBoost, Support Vector Machines (SVM), Random Forest, Naive Bayes, Gradient Boosting, and Decision Tree 
- **Advanced Modeling**: LSTM, GRU, CNN, and BERT-based topic modeling  
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
1. XGBoost and CNN achieved the highest accuracy, demonstrating strong generalization.
2. Deep learning methods are effective for understanding textual patterns, but combining them with traditional ML can yield optimal results.
3. Feature representation (like TF-IDF or Word2Vec) significantly affects model performance.


  
## Reflection
This project provided a hands-on opportunity to explore the intersection of NLP and classification models. Understanding how different techniques interpret textual information, and evaluating their effectiveness, was a valuable learning experience in real-world text mining applications.


## Author
Developed by **Araa Almarhabi** as part of the **Natural Language Processing** course.







