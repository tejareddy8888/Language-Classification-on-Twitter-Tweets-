#!/usr/bin/env python
# coding: utf-8

#%%
from __future__ import division, print_function, unicode_literals
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# # Taking the data input and then merge them 
workspace_path = r'C:\Users\Govur\Documents\Class\uzh\Fall_2019\Machine Learning for NLP\exercises\exercise_1'
file_path = os.path.join(workspace_path, "labels-train+dev.tsv")
train_df = pd.read_csv(file_path, sep = '\t' , encoding = 'utf-8',header=None,names=['Label','TweetID'])
#train_df.info()

file_path = os.path.join(workspace_path, "labels-test.tsv")
test_df = pd.read_csv(file_path, sep = '\t' , encoding = 'utf-8',header=None,names=['Label','TweetID'])
#test_df.head()

file_path = os.path.join(workspace_path, "tweets.json")
tweets_df = pd.read_json(file_path,encoding = 'utf-8',orient = 'values',dtype='int64',lines=True)

tweets_df.rename(columns={0: "TweetID", 1: "Tweets"}, inplace=True)
#tweets_df.info()

train_df = pd.merge(train_df, tweets_df, on='TweetID')
test_df = pd.merge(test_df, tweets_df, on='TweetID')

Opt_train_df= train_df.groupby('Label').filter(lambda x : len(x)>10)
print(Opt_train_df['Label'].nunique())
Opt_train_df.reset_index(inplace=True)

Opt_test_df = test_df[test_df['Label'].isin(Opt_train_df['Label'])]
print(Opt_test_df['Label'].nunique())
Opt_test_df.reset_index(inplace=True)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.1,train_size=0.9 ,random_state=19)
for train_index, test_index in split.split(Opt_train_df, Opt_train_df["Label"]):
    strat_train_set = Opt_train_df.loc[train_index]
    strat_test_set = Opt_train_df.loc[test_index]


x_trainingtweet = Opt_train_df.Tweets
y_trainingtweet = Opt_train_df.Label
x_trainingtweet_dev = strat_train_set.Tweets.astype('str')
y_trainingtweet_dev = strat_train_set.Label.astype('str')
x_trainingtweet_val = strat_test_set.Tweets.astype('str')
y_trainingtweet_val = strat_test_set.Label.astype('str')
x_testingtweet = Opt_test_df.Tweets
y_testingtweet = Opt_test_df.Label

print('x_trainingtweet set size is ',  x_trainingtweet.shape)
print('y_trainingtweet set size is ', y_trainingtweet.shape)
print('x_trainingtweet_dev set size is ',  x_trainingtweet_dev.shape)
print('y_trainingtweet_dev set size is ',  y_trainingtweet_dev.shape)
print('x_trainingtweet_val set size  is ',  x_trainingtweet_val.shape)
print('y_trainingtweet_val set size  is ',  y_trainingtweet_val.shape)
print('x_testingtweet set size  is ',  x_testingtweet.shape)
print('y_testingtweet set size  is ',  y_testingtweet.shape)

label_encoder = LabelEncoder()
y_dev_trainingtweet = label_encoder.fit_transform(y_trainingtweet_dev)
y_val_trainingtweet = label_encoder.transform(y_trainingtweet_val)
y_testingtweet = label_encoder.transform(y_testingtweet)


CountVector = CountVectorizer(ngram_range=(1,3),analyzer='char_wb')
training_dev_data_count = CountVector.fit_transform(x_trainingtweet_dev)
training_val_data_count = CountVector.transform(x_trainingtweet_val)
testing_data_count = CountVector.transform(x_testingtweet)

#%%
'''
TfidfVector = TfidfVectorizer(ngram_range=(3,3),analyzer='char_wb')
training_data_count = TfidfVector.fit_transform(x_trainingtweet)
testing_data_count = TfidfVector.transform(x_testingtweet)
'''

tfidf=TfidfTransformer(smooth_idf=False)
train_dev_data_tfidf = tfidf.fit_transform(training_dev_data_count)
train_val_data_tfidf = tfidf.transform(training_val_data_count)
test_data_tfidf = tfidf.transform(testing_data_count)
#print(train_data_tfidf[0,:])




#Creating pipeline

text_nbclf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3),analyzer='char_wb')),('nb_clf', MultinomialNB(fit_prior=False))]) 
text_nbclf.fit(x_trainingtweet_dev,y_dev_trainingtweet)
nb_scores = cross_val_score(text_nbclf, x_trainingtweet_dev, y_dev_trainingtweet, scoring='accuracy', cv=10)


print('Accuracy of test set size  is %.6f'%text_nbclf.score(x_testingtweet, y_testingtweet))
print('Accuracy of Development set size  is %.6f'%text_nbclf.score(x_trainingtweet_dev, y_dev_trainingtweet))
print('Accuracy of validation set size  is %.6f'%text_nbclf.score(x_trainingtweet_val, y_val_trainingtweet))



print(precision_score(y_testingtweet,text_nbclf.predict(x_testingtweet), average='micro'))
print(recall_score(y_testingtweet,text_nbclf.predict(x_testingtweet), average='micro'))
print(f1_score(y_testingtweet,text_nbclf.predict(x_testingtweet), average='weighted'))

conf_mx = confusion_matrix(y_testingtweet,text_nbclf.predict(x_testingtweet))
print(conf_mx)


import matplotlib.pyplot as plt

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# Schostic with term frequency-inverse document frequency
text_sgdclf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3),analyzer='char_wb')),('tfidf', TfidfTransformer(smooth_idf=False)),('sg_train', SGDClassifier(loss='modified_huber',penalty='l2',fit_intercept=False))]) 
text_sgdclf.fit(x_trainingtweet_dev,y_dev_trainingtweet)
sgd_scores = cross_val_score(text_sgdclf, x_trainingtweet_dev, y_dev_trainingtweet, scoring='accuracy', cv=10)

print('Accuracy of TEST set size  is %.6f'%text_sgdclf.score(x_testingtweet, y_testingtweet))
print('Accuracy of DEVELOPMENT  set size  is %.6f'%text_sgdclf.score(x_trainingtweet_dev, y_dev_trainingtweet))
print('Accuracy of VALIDATION set size  is %.6f'%text_sgdclf.score(x_trainingtweet_val, y_val_trainingtweet))

print('precision score is %.6f'%precision_score(y_testingtweet,text_sgdclf.predict(x_testingtweet), average='weighted'))
print('recall score  is %.6f'%recall_score(y_testingtweet,text_sgdclf.predict(x_testingtweet), average='weighted'))
print('f1score  is %.6f'%f1_score(y_testingtweet,text_sgdclf.predict(x_testingtweet), average='micro'))
conf_mx = confusion_matrix(y_testingtweet,text_sgdclf.predict(x_testingtweet))
#print(conf_mx)


import matplotlib.pyplot as plt

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()



param_grid = {'vect__ngram_range': [(1,3),(2,3)],
'vect__analyzer': ['char', 'char_wb'],
'sg_train__loss': ['hinge', 'modified_huber'],
'sg_train__penalty': ['none', 'l2', 'elasticnet'],
'sg_train__class_weight' : ['balanced']}

gs_scv = GridSearchCV(text_sgdclf, param_grid, cv=2, n_jobs=15, verbose=1)
gs_scv.fit(x_trainingtweet_dev, y_dev_trainingtweet)


gscv_df = pd.DataFrame(gs_scv.cv_results_)
gscv_df.sort_values(by=['rank_test_score'])
print(precision_score(y_testingtweet,gs_scv.predict(x_testingtweet), average='micro'))
print(recall_score(y_testingtweet,gs_scv.predict(x_testingtweet), average='micro'))
print(f1_score(y_testingtweet,gs_scv.predict(x_testingtweet), average='micro'))
conf_mx = confusion_matrix(y_testingtweet,gs_scv.predict(x_testingtweet))
conf_mx

import matplotlib.pyplot as plt

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

