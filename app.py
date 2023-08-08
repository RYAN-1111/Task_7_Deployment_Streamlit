import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline 
import openai
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from textblob import TextBlob
import cleantext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
from scrapingbee import ScrapingBeeClient

# Configure your OpenAI API key
openai.api_key = "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"


def generate_chatbot_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response['choices'][0]['text'].strip()

def sentiment_analysis():
        
    with st.expander('Analyze Text'):
        text = st.text_input('Text here: ')
        if text:
            blob = TextBlob(text)
            if round(blob.sentiment.polarity,2) >=0.5:
                st.write('Positive')
            elif round(blob.sentiment.polarity,2) <=-0.5:
                st.write('Negative')
            else:
                st.write('Neutral')
        
def scrape_amazon_product_page(url):
    headers = {"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    client = ScrapingBeeClient(api_key='92R2GCY78T0X6IZALLQI0H5KBXIB7GFJ8Y0NZ7MKNBM0PMKPDWCSH821J0MHNKD9YDH9SZD7O8O81OTE')
    page = client.get(url)
    soup1 = BeautifulSoup(page.content, 'html.parser')
    soup2 = BeautifulSoup(soup1.prettify(), "html.parser")
    product = soup2.find('span', {'id': 'productTitle'})
    product_name = product.get_text().strip() if product else ''
    category_element = soup2.find('a', {'class': 'a-link-normal a-color-tertiary'})
    category = category_element.get_text().strip() if category_element else ''
    description_element = soup2.find('div', {'name': 'description'})
    description = description_element.get_text().strip() if description_element else ''
    price_element = soup2.find('span', 'a-offscreen')
    price = price_element.get_text().strip() if price_element else ''

    reviews = []
    review_elements = soup2.find_all('span', {'class': 'a-size-base review-text'})
    for review_element in review_elements:
        reviews.append(review_element.get_text().strip())

    rating_element = soup2.find('span', {'class': 'a-icon-alt'})
    rating = rating_element.get_text().strip() if rating_element else ''

    data = {
        'Product Name': [product_name],
        'Category': [category],
        'Description': [description],
        'Price': [price],
        'Reviews': ['\n'.join(reviews)],
        'Rating/Specifications': [rating]
    }
    df = pd.DataFrame(data)
    return df

def chatbot(product_data):
    st.subheader("Chatbot")
    user_input = st.text_input("Ask a question or provide an inquiry:")
    
    if st.button("Chat"):
        if user_input.strip() != "":
            product_info = f"Product Name: {product_data['Product Name'][0]}\nCategory: {product_data['Category'][0]}\nPrice: {product_data['Price'][0]}\nDescription: {product_data['Description'][0]}"
            prompt = f"You: {user_input}\nProduct Info:\n{product_info}\nChatbot:"
            response = generate_chatbot_response(prompt)
            st.write("Chatbot: " + response)

def main():
    st.title("Sentiment Analysis & Chatbot")

    st.sidebar.header("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Sentiment Analysis", "Chatbot"])

    if selected_page == "Sentiment Analysis":
        sentiment_analysis()
    elif selected_page == "Chatbot":
        product_url = "https://www.amazon.in/dp/B0BJ72WZQ7?ie=UTF8&viewID=&ref_=s9_acss_bw_cg_halo_3b1_w"
        product_data = scrape_amazon_product_page(product_url)  # Perform web scraping
        chatbot(product_data)  # Pass product data to the chatbot function

if __name__ == "__main__":
    main()
