# Installing Libraries
import numpy as np
import pandas as pd
import nltk

# Ensure both punkt and punkt_tab are available
for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


stemmer=PorterStemmer()

# Tokenize Sentences (breaking sentence into words)
def tokenize(sentence):
    return word_tokenize(sentence)

# Stemming (converting into root words)
def stemming(words):
    return stemmer.stem(words.lower())

# Bag of words return array of 0 (if corresponding word of words array not present in tokenized sentence) or 1(if present) in their corresponding position
def bag_of_words(tokenized_sentences,words):
    bag=np.zeros(len(words),dtype=np.float32)
    tokenized_words=[]
    stemmed_words=[]
    for w in tokenized_sentences:
        tokenized_words.append(stemming(w))
    for w in words:
        stemmed_words.append(stemming(w))
    for index,w in enumerate(stemmed_words):
        if w in tokenized_words:
            bag[index]=1
            
    return bag        



