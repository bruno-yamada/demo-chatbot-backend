
# coding: utf-8

import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

print('Downloading nltk packages')
nltk.download('rslp')
nltk.download('stopwords')
stemmer = nltk.stem.RSLPStemmer()
stopwords = nltk.corpus.stopwords.words('portuguese')

data = pd.read_csv('data.csv', delimiter=';', header=0)

# clf = LinearSVC()
# clf = AdaBoostClassifier()
clf = MultinomialNB()
# clf = BernoulliNB()
# clf = RidgeClassifier()
    



def stemmize(text):
    words = []
    for w in text.split():
        words.append(stemmer.stem(w))
    return ' '.join(words)

def remove_stopwords(text):
    words = []
    for w in text.split():
        if w not in stopwords:
            words.append(w)
    return ' '.join(words)



encoder = LabelEncoder()
vectorizer = TfidfVectorizer()
data['frase'] = data['frase'].apply(remove_stopwords)
data['frase'] = data['frase'].apply(stemmize)
X, y = data['frase'], data['intencao']
X = vectorizer.fit_transform(data['frase'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Classificador com acerto de {0:.1f}%".format(score*100))



def predict(text):
    text = remove_stopwords(text)
    text = stemmize(text)
    vect = vectorizer.transform([text])[0]
    predictions = [{'intencao': k, 'prob':v} for k,v in zip(clf.classes_, clf.predict_proba(vect)[0])]
    return list(sorted(predictions, key=lambda k: k['prob'], reverse=True))


while True:
    print('-'*10)
    print('Mensagem:')
    text = input()
    predicao = predict(text)
    print('> Predição:', predicao[0])
