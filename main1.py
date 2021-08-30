# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 08:11:08 2020

@author: mohit
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc

import re
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
            

##### Part 1  :   Basic Statistics ########

#reading train.csv file ....

df = pd.read_csv("train.csv")
print("number of points ",df.shape[0])

df.head()

df.groupby("is_duplicate")["id"].count().plot.bar()

#Percentage of similar and dis-similar points ....

print("Duplicate Question Percentage : ", round(df['is_duplicate'].mean()*100,2))
print("Non - Duplicate Question Percentage : ", 100-round(df['is_duplicate'].mean()*100,2))

#No of unique and repeated Questions ..

qids = pd.Series(df['qid1'].to_list()+df['qid2'].to_list())
unique_qs = len(np.unique(qids))
qs_moreThanOnce = np.sum(qids.value_counts() > 1)

print("No. of unique Questions : ",unique_qs)
print("No. of repeated Questions : ",qs_moreThanOnce)

x = ["Unique Questions", "Repeated Questions"]
y = [unique_qs,qs_moreThanOnce]
plt.figure(figsize = (10,6))
plt.title("Plot Representing Unique and Repeated Questions")
sns.barplot(x,y)
plt.show()

#checking for null values and Replacing them with ' '
NullRows = df[df.isnull().any(1)]
df = df.fillna('')
nan_rows = df[df.isnull().any(1)]
print(nan_rows)

#Feature Extration (Entry-level) ..........
#i have defined the features to be :
 
# fqid1/2 = frequency of qid's in the dataset
# q1_len,q2_len = length of both the questions across all rows...
# q1_words,q2_words = no. of words in q1 and q2:
# Common_words = Common words bw q1 and q2




df['fqid1'] = df.groupby('qid1')['qid1'].transform('count')
df['fqid2'] = df.groupby('qid2')['qid2'].transform('count')
df['q1_len'] = df['question1'].str.len()
df['q2_len'] = df['question2'].str.len()
df['q1_words'] = df['question1'].apply(lambda row : len(row.split(" ")))
df['q2_words'] = df['question2'].apply(lambda row : len(row.split(" ")))


def common_words(row):
    w1 = set(map(lambda word : word.lower().strip() ,row['question1'].split(" ")))
    w2 = set(map(lambda word : word.lower().strip() ,row['question2'].split(" ")))
    return len(w1 & w2)
df["Common_words"]= df.apply(common_words,axis = 1)

def total_words(row):
    w1 = set(map(lambda word : word.lower().strip() , row['question1'].split(" ")))
    w2 = set(map(lambda word : word.lower().strip() , row['question2'].split(" ")))
    return len(w1) + len(w2)
df["Total Words"] = df.apply(total_words,axis = 1)
 
def normalized_word_share(row):
      w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
      w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
      return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

df['word_share'] = df.apply(normalized_word_share, axis=1)
df['freq_q1+q2'] = df['fqid1']+df['fqid2']
df['freq_q1-q2'] = abs(df['fqid1']-df['fqid2'])

#Now lets Visualize the feature word_share against independant variable:

plt.figure(figsize = (12,8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate',y = 'word_share', data = df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_share'][0:],label = "1",color = 'red' )
sns.distplot(df[df['is_duplicate'] == 0.0]['word_share'][0:],label = "0",color = 'blue')

#Now lets Visualize the feature Common_words against independant variable

plt.figure(figsize = (12,8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate',y = 'Common_words',data = df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['Common_words'][0:],label = "1",color = 'red')
sns.distplot(df[df['is_duplicate'] == 0.0]['Common_words'][0:],label = "0",color = 'blue')

# As evident from plots of both the features , word_share seems to carry more value than Common_word
# as there is less overlap in its distribution, and in the plot of word_share, its observed that 
# as the word_share increases , questions tend to be more duplicate, which is a useful info in itself.

#On the other hand , The distributions of Common_word are highly overlapping, nothing can be said with 
#Certainity after observing their plots...




 


