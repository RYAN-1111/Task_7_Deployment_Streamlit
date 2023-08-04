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

# Configure your OpenAI API key
openai.api_key = "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"

# # Load and preprocess your data for sentiment analysis
# df = pd.read_csv('merged_dataset.csv')
# df.dropna(inplace=True)
# blanks = [i for i, lb, rv in df.itertuples() if type(rv) == str and rv.isspace()]
# df.drop(blanks, inplace=True)

# # Split data into train and test sets
# X = df['review']
# y = df['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Train the sentiment analysis model
# tfidf_vectorizer = TfidfVectorizer(max_features=5000)
# X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# X_test_tfidf = tfidf_vectorizer.transform(X_test)

# sentiment_model = LinearSVC()
# sentiment_model.fit(X_train_tfidf, y_train)

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
        



        

            

        

def chatbot():
    # st.header("Chatbot")
    # st.write("Welcome to the Product Inquiry Chatbot!")
    # st.write("Type 'exit' to end the conversation.")

    # while True:
    #     user_input = st.text_input("You:", "")

    #     if user_input.lower() == 'exit':
    #         st.write("Chatbot: Goodbye!")
    #         break

    #     # Process user input and generate chatbot response
    #     prompt = f"You: {user_input}\nChatbot:"
    #     chatbot_response = generate_chatbot_response(prompt)

    #     st.write(f"Chatbot: {chatbot_response}")
    st.subheader("Chatbot")
    user_input = st.text_input("Ask a question or provide an inquiry:")
    if st.button("Chat"):
        if user_input.strip() != "":
            response = generate_chatbot_response(user_input)
            st.write("Chatbot: " + response)


def main():
    st.title("Sentiment Analysis & Chatbot")

    st.sidebar.header("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Sentiment Analysis", "Chatbot"])

    if selected_page == "Sentiment Analysis":
        sentiment_analysis()
    elif selected_page == "Chatbot":
        chatbot()

if __name__ == "__main__":
    main()