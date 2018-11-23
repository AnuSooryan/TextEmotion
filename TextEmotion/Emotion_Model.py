# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:06:55 2018

@author: Anu
"""

import pandas as pd
from .preprocess import clean_text, label
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import os

path = os.getcwd() + '/TextEmotion/TextEmotion/'
df = pd.read_csv(path + 'data.csv')

df = df.replace({"sentiment": label})

df['content'] = df['content'].map(lambda x: clean_text(x))
df= df.dropna()

clf = Pipeline([('vectorizer', TfidfVectorizer()), 
                ('classifier', RandomForestClassifier(n_estimators=100)),
                ])
clf.fit(df['content'], df['sentiment'])
joblib.dump(clf, 'model.pkl')
