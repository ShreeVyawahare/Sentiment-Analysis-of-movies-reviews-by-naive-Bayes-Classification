import numpy as np
import pandas as pd

dataset = pd.read_csv('C:\machine learning\sentiment analysis of movies review\IMDB Dataset.csv')
dataset.head()

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
docs = np.array(['Hello I am shrirang vyawahare'])
bag = vectorizer.fit_transform(docs)
print(vectorizer.vocabulary_)
                 
print(bag.toarray())

from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(precision=2)
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(bag).toarray())

import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(use_idf = True,norm = 'l2',smooth_idf = True)
y = dataset.sentiment.values
x = tfidf.fit_transform(dataset['review'].values.astype('U'))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)