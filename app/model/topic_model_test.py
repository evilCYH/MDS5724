# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:22:27 2024

@author: Neal
"""

import joblib
from topic_model_dev import jieba_tokenizer
from sklearn.metrics import f1_score, confusion_matrix

test_data = ["苏农银行:苏农银行第六届董事会第十四次会议决议公告",
             "辉丰股份:独立董事候选人声明与承诺-杨兆全",
             "华脉科技:华脉科技2022年年度报告",
             "百济神州:百济神州有限公司章程",
             "莱宝高科:2022年年度权益分派实施公告"]



pipeline = joblib.load('topic_classification_model_v003.joblib')
print(pipeline.predict(test_data))
