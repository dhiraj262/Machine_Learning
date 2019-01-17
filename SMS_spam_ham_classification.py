# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_table('SMSSpamCollection',header = None, encoding = 'utf-8')

dataset[0].value_counts()
""" 
ham     4825
spam     747
Name: 0, dtype: int64

"""
# Data Processing

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(dataset[0])

messages = dataset[1]

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = messages.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')

# Replace email addresses with 'email'
processed = processed.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')
    
# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# change words to lower case - Hello, HELLO, hello are all the same word
processed = processed.str.lower()
corpus = []

for i in range(0,len(processed)):
    message = re.sub(r'\W',' ',str(processed[i]))
    message = re.sub(r'\s+[a-z]\s',' ',message)
    message = re.sub(r'^\s+|\s+?$',' ',message)
    message = re.sub(r'[^\w\d\s]',' ',message)
    message = re.sub(r'\s+',' ',message)
    corpus.append(message)
    
# Generating features
from nltk.tokenize import word_tokenize

# create bag-of-words
all_words = []

for message in corpus:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)

print('Most common words: {}'.format(all_words.most_common(50)))

wordcloud = WordCloud().generate(str(corpus))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(max_features= 2000, stop_words=stopwords.words('english'))
X = vec.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_eval,y_train,y_eval  = train_test_split(X,y,test_size = 0.25, random_state = 0)

model = SVC(kernel = 'linear')  
model.fit(X_train,y_train) 

names_classifiers = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names_classifiers, classifiers)

for name, model in models:
    model.fit(X_train,y_train)
    pred = model.predict(X_eval)

    cm = confusion_matrix(y_eval,pred)
    print("{} Accuracy: {}".format(name,(np.trace(cm))/(np.sum(cm))))
    print(classification_report(y_eval,pred))



    