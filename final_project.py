#################### Importing dataset and required libraries ###########################
import pandas as pd
import numpy as np
import matplotlib as plt
from pandas import read_csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder 
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import re
from nltk.corpus import stopwords


df=read_csv("../input/zomato-bangalore-restaurants/zomato.csv")



####################### Pre-processing the data ###################################
print(type(df))
print(len(df))

#Dropping columns
new=df.drop(["url","phone","dish_liked","menu_item"],axis=1)

#Converting incosistent values to NA and dropping rows
new[new=='']= np.nan
new[new=='NEW']= np.nan
new[new=='-']= np.nan
new[new=='[]']= np.nan
new1 = new.dropna()
print(len(new1))

#Some more preprocessing
new1['rate']=new1['rate'].replace("/5","", regex=True)
new1['rate']=new1['rate'].astype('float32')

new1.book_table[new1.book_table=="Yes"]=1
new1.book_table[new1.book_table=="No"]=0

new1.online_order[new1.online_order=="Yes"]=1
new1.online_order[new1.online_order=="No"]=0

new1.name= new1.name.map(lambda x: x.encode('ascii','ignore').decode('ascii'))

new1['approx_cost(for two people)']=new1['approx_cost(for two people)'].replace(",","", regex=True)
new1['approx_cost(for two people)']=new1['approx_cost(for two people)'].astype('float32')

#Reindexing dataset
new1.index= range(len(new1))
