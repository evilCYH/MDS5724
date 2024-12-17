# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:35:51 2024

@author: Neal
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pandas as pd
import jieba
import joblib


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
    words = list(jieba.cut(text))
    return words



nb_clf = MultinomialNB()
lr_clf = LogisticRegression(max_iter=1000)
svc_clf = LinearSVC(max_iter=1000)


pipeline = Pipeline([
    ('vect', TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)),
    ('clf', lr_clf),  
])


parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'vect__max_df': [0.8, 0.9],
    'vect__min_df': [2, 5],
    'vect__use_idf': [True, False],
    'vect__norm': ['l1', 'l2'],
    'clf': [nb_clf, lr_clf, svc_clf]  
}


nb_params = {
    'clf__alpha': [0.1, 1.0],
    'clf__fit_prior': [True, False]
}

lr_params = {
    'clf__C': [0.1, 1.0, 10.0],
    'clf__penalty': ['l1', 'l2']
}

svc_params = {
    'clf__C': [0.1, 1.0, 10.0],
    'clf__loss': ['squared_hinge']
}


def get_params_for_classifier(clf):
    """根据分类器类型返回对应的参数"""
    if isinstance(clf, MultinomialNB):
        return nb_params
    elif isinstance(clf, LogisticRegression):
        return lr_params
    elif isinstance(clf, LinearSVC):
        return svc_params
    return {}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block


    grid_search = GridSearchCV(
        pipeline, 
        parameters, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
        cv=5
    )

    df = pd.read_excel(r"D:/OneDrive/Desktop/Task-2/train_samples_20000.xlsx")

    print(f"Process {len(df.news_title)} titles and {len(df.topic)} categorical labels")
    grid_search.fit(df.news_title, df.topic)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    
  
    best_clf = best_parameters['clf']

    best_clf_params = get_params_for_classifier(best_clf)
    
    for param_name in sorted(parameters.keys()):
        if param_name in best_parameters:
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    

    for param_name in sorted(best_clf_params.keys()):
        if param_name.replace('clf__', '') in best_parameters:
            print("\t%s: %r" % (param_name, best_parameters[param_name.replace('clf__', '')]))

    joblib.dump(grid_search, "topic_classification_model_v003.joblib")