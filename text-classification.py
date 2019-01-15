import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from nltk.tokenize import word_tokenize

# IMPORTING THE DATASET
sentiment = load_files('txt_sentoken/')
X,y = sentiment.data, sentiment.target

#print(X,y)

corpus = []

for i in range(0,len(X)):
    sentiment = re.sub(r'\W',' ',str(X[i]))
    sentiment = sentiment.lower()
    sentiment = re.sub(r'\s+[a-z]\s',' ',sentiment)
    sentiment = re.sub(r'\s+',' ',sentiment)
    corpus.append(sentiment)

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(max_features= 2000,min_df= 2,max_df=0.5, stop_words=stopwords.words('english'))
X = vec.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_eval,y_train,y_eval  = train_test_split(X,y,test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

pred = classifier.predict(X_eval)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_eval,pred)

accuracy = (np.trace(cm))/(np.sum(cm))
print(accuracy)