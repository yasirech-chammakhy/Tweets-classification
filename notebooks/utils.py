# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import contractions
from sklearn.feature_extraction.text import CountVectorizer

# Function to clean the tweet text
def clean_tweets(text):
    """
    The function `clean_tweets` takes in a text and performs various preprocessing steps such as
    removing URLs, usernames, digits, expanding contractions, converting to lowercase, tokenizing,
    removing punctuation, lemmatizing, and removing stopwords. The function then returns the cleaned
    text as a string.
    
    :param text: The input text that you want to clean and preprocess
    :return: a cleaned version of the input text. The text is processed to remove URLs, usernames,
    digits, and punctuation. Contractions are expanded and the text is converted to lowercase. Then,
    NLTK preprocessing steps are applied, including tokenization, lemmatization, and removal of
    stopwords. The final result is a string of cleaned words joined together.
    """
    # Remove URLs
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    # Remove usernames and keep the remaining text
    text = re.sub(r'@([A-Za-z0-9_]+)', r'\1', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Expand contractions
    text = contractions.fix(text)
    # Convert to lowercase
    text = text.lower()

    # NLTK preprocessing steps
    lemmatizer = WordNetLemmatizer()
    tokenizer = TweetTokenizer()
    stop_words = set(stopwords.words('english'))

    # Tokenize
    words = tokenizer.tokenize(text)
    # Remove punctuation
    words = [re.sub(r'[^\w\s]', '', word) for word in words if word != '']
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words).strip()


# function to get the most common words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# function to plot the most common words
def plot_top_n_words(corpus, n=None, title=None):
    common_words = get_top_n_words(corpus, n)
    df = pd.DataFrame(common_words, columns=['word', 'count'])
    plt.figure(figsize=(10, 5))
    sns.barplot(x='count', y='word', data=df)
    plt.title(title)
    plt.show()
