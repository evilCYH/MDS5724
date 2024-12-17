# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:35:51 2024

@author: Neal
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import jieba
import joblib


# 中文分词器
def jieba_tokenizer(text):
    """
    Customize our own tokenizer for Chinese with jieba 
    Parameters
    ----------
    text : STR
        raw text.

    Returns
    -------
    words : List 
        List of cutted words.

    """
    # 使用jieba进行分词
    words = list(jieba.cut(text))
    return words



# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier

# 管道（需要优化的地方）
pipeline = Pipeline([
    # 使用CountVectorizer提取文本词频特征
    ('vect', CountVectorizer(tokenizer = jieba_tokenizer, token_pattern=None)),
    # 将词频转成TFIDF矩阵
    ('tfidf', TfidfTransformer()),
    # 使用多项式贝叶斯分类器
    ('clf', MultinomialNB()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way

#  n-gram范围（unigram bigram），是否使用tfidf权重
parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'tfidf__use_idf': (True, False),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    
    grid_search = GridSearchCV(pipeline, parameters, scoring='f1_weighted', n_jobs=-1, verbose=1)
    # df = pd.read_excel('./data/Task-2/train.xlsx')

    df = pd.read_excel(r"D:/OneDrive/Desktop/Task-2/train_samples_20000.xlsx")
    # ./data/train_samples_20000.xlsx

    print(f"Process {len(df.news_title)} titles and {len(df.topic)} categorical labels")

    grid_search.fit(df.news_title, df.topic)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    joblib.dump(grid_search, "topic_classification_model_v001.joblib")