
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


label_encoder = LabelEncoder()
y_dev_trainingtweet = label_encoder.fit_transform(y_trainingtweet_dev)
y_val_trainingtweet = label_encoder.transform(y_trainingtweet_val)
y_testingtweet = label_encoder.transform(y_testingtweet)

text_mlp = Pipeline([('vect', CountVectorizer(ngram_range=(1,3),analyzer='char_wb')),('tfidf', TfidfTransformer(smooth_idf=True)),('mlp_clf', MLPClassifier(hidden_layer_sizes=(100,7),max_iter=100,learning_rate='adaptive',learning_rate_init=0.004))]) 
text_mlp.fit(x_trainingtweet_dev, y_dev_trainingtweet)

mlp_scores = text_mlp.score(x_testingtweet, y_testingtweet)
print('Accuracy of Development set size  is %.6f'%text_mlp.score(x_trainingtweet_dev, y_dev_trainingtweet))
print('Accuracy of validation  set size  is %.6f'%text_mlp.score(x_trainingtweet_val, y_val_trainingtweet))
print('Accuracy of Test  set size  is %.6f'%mlp_scores)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
print(f1_score(y_testingtweet,text_mlp.predict(x_testingtweet), average='weighted'))

conf_mx = confusion_matrix(y_testingtweet,text_mlp.predict(x_testingtweet))
conf_mx


import matplotlib.pyplot as plt
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


