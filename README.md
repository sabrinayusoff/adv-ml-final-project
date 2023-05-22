# CAPP30255 Advanced Machine Learning Final Project
### Authors: Phoebe Collins and Sabrina Yusoff

This project aims to build a machine learning model to answer the question: "Can you win Jeopardy! using machine learning?"
Using a dataset of 200,000 questions aired on Jeopardy between 2004-2006, we seek to build a model that predicts four outputs: Category, Value, Round, and Answer.
The project comprises 4 key stages:

### 1. Baseline model: Logistic Regression with BoW features
This is a multiclass classification model converting textual data from the Jeopardy questions into
Bag-of-Words (BoW) feature vectors to predict Round, Value and Category.
Files:  
* final_notebooks/logistic_regression_categories_bow.ipynb
* final_notebooks/logistic_regression_values_bow.ipynb
* final_notebooks/logistic_regression_round_bow.ipynb

### 2. Baseline model: Logistic Regression with TF-IDF features
Given that textual data can be pre-processed in many ways, we examined whether representing the questions
as TF-IDF vectors would yield better performance.
Files:  
* final_notebooks/logistic_regression_categories_tfidf.ipynb
* final_notebooks/logistic_regression_values_tfidf.ipynb
* final_notebooks/logistic_regression_round_tfidf.ipynb

### 3. Fine-tuned DistillBERT model
Models that have been pre-trained on huge natural language datasets can offer much better performance when finetuned on specific tasks.
In our case, we finetuned the DistillBERT model on the Jeopardy! dataset and obtained significantly higher
prediction accuracy.
Files:  
* final_notebooks/round_classification_even.ipynb
* final_notebooks/round_classification_uneven.ipynb
* final_notebooks/value_classification_double_jeopardy.ipynb
* final_notebooks/value_classification_jeopardy.ipynb
* final_notebooks/value_classification_jeopardy_even.ipynb
* final_notebooks/value_classification_jeopardy_uneven.ipynb
* final_notebooks/category_classification.ipynb

### 4. Generative Question-Answering using Retrieval-Augmented Generation (RAG)
Given the lack of contextual information accompanying Jeopardy! questions, traditional question-answer models cannot
be used to predict answers to these questions. We explored the RAG model which allows generative question-answering by combining a pre-trained dense retrieval (DPR) and sequence-to-sequence (seq2seq) models.
Files:  
* final_notebooks/qna.ipynb


Heatmap plots from Stages 1, 2 and 3 are found in the plots/ folder.