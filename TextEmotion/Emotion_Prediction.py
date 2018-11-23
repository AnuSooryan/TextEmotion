# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:22:02 2018

@author: Anu
"""


from .preprocess import clean_text, label
from sklearn.externals import joblib

def prediction(text):
    model = joblib.load('model.pkl')
    text = clean_text(text)
    predicted = model.predict([text])
    predicted = int(predicted[0])
    prediction = [k for k, v in label.items() if v == predicted][0]
    return prediction