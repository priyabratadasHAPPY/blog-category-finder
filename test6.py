# !pip install streamlit transformers

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax

# Load the model and tokenizer
MODEL = "priyabrat/new5th_bert_article_categorisation_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Define the categorization function
def categorize_blog(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512, add_special_tokens=True)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores *= 100  # Convert to percentages
    # Round the scores to two decimal places
    scores = np.round(scores, 2)
    categories = ['Health', 'Sports', 'Travel', 'Food', 'Fashion', 'Education', 'Finance', 'Political']
    # Combine categories and scores, and sort by score descending
    category_scores = sorted(zip(categories, scores), key=lambda x: x[1], reverse=True)
    return category_scores

# Streamlit app
st.title("Blog Content Categorization Tool for Writers")

# Subtitle and explanation
st.header("Identify Your Blog's Category")
st.write("This tool helps blog content writers to categorize their articles based on content. Simply paste your blog text below and hit 'Categorize'.")

# Text area for user input
text = st.text_area("Paste your blog content here:")

if st.button('Categorize'):
    # Categorize the blog content
    category_scores = categorize_blog(text)
    st.write("Categorization Results:")
    for category, score in category_scores:
        st.write(f"{category}: {score}%")
