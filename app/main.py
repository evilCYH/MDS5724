# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:12:48 2024

@author: Neal
"""
from pydantic import BaseModel
import model.topic_model as clf
from fastapi import FastAPI
from joblib import load
from model.topic_model_dev import jieba_tokenizer
import jieba
import uvicorn
        
app = FastAPI(title="MDS5724 Group Project - Task2 - Demo", 
              description="API for Text Classification", version="0.001")

class Payload(BaseModel):
    news_title: str = ""

@app.on_event('startup')
def load_model():
    jieba.initialize()
    clf.model = load('model/topic_classification_model_v001.joblib')


@app.post('/predict')
async def get_prediction(payload: Payload = None):
    news_title = dict(payload)['news_title']
    score = clf.model.predict([news_title,]).tolist()[0]
    return score

if __name__ == '__main__':
    uvicorn.run(app, port=5724, host='0.0.0.0')