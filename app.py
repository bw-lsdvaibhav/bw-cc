# Prepare libraries
from flask import Flask, render_template, request
import pickle
import copy
import nltk
import re
import heapq  
import pickle
import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
punctuation = punctuation + '\n'
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

# Stop not important warnings and define the main flask application
warnings.filterwarnings("ignore")
app = Flask(__name__)

def prepare_data(text):
    text = re.sub(r"([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))+",' ',text)
    text = re.sub(r"\d", ' ', text)
    text = re.sub(r"\t+",' ',text)
    text = re.sub(r"\n+",' ',text)
    text = re.sub(r"\W+",' ',text)
    text = re.sub(r"^ +",'',text)
    text = re.sub(r" +$",'',text)
    text = re.sub(r" +",' ',text)
    text = text.capitalize()
    
    return text
# Delete stopwords:
# Like prepositions and hyphens words. for example [**and, in, or ...etc**].
def delete_stopwords(input_text):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(input_text)
    wnl = nltk.WordNetLemmatizer()
    lemmatizedTokens = [wnl.lemmatize(t) for t in tokens]
    out_text = [w for w in lemmatizedTokens if w not in stop_words]
    out_text = ' '.join(out_text)
    return out_text
def preprocessing(text):
    text = str(text)
    text = prepare_data(text)
    text = delete_stopwords(text)
    return text

# Application home page
@app.route("/")
def index():
    return render_template("index.html", page_title="Product & Categorization")

@app.route("/analyze_text", methods=['GET', 'POST'])
def analyze_text():
    if request.method == 'POST':
        input_text = request.form['text_input_text']
        classifier_model_name = request.form['text_classifier']
        input_text = preprocessing(input_text)
        tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pickle", "rb"))
        test_features = tfidf_vectorizer.transform([input_text])
        model = pickle.load(open('en_'+classifier_model_name+'.pkl','rb'))
        text_predection = model.predict(test_features.toarray())
        
    return render_template("index.html", page_title="Career Mapping", input_text=input_text, text_category=text_predection)

# Start the application on local server
if __name__ == "__main__":
    app.run()
