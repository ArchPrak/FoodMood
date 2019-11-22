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

##################################### EDA #############################################################################3
#Correlogram
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(new1.corr(), xticklabels=new1.corr().columns, yticklabels=new1.corr().columns, cmap='RdYlGn', center=0, annot=True)

plt.title('Correlogram ', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#Printing contingency table
contingency_table=pd.crosstab(data["book_table"],data["online_order"])
print('contingency_table :-\n',contingency_table)


#Wordcloud
data=df

	#Finding most popular location
	#pop_location="Banashankari"
pop_location=data.mode()["location"]

	#Filtering to gwt rows with that location
d=data.loc[data["location"] == pop_location[0]]

l=[]
for i in list(d["dish_liked"]):
    if type(i)==str:
        l.extend(i.split(","))

text=" ".join(l) 

	#Create the wordcloud object
wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)
 
	#Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()



#Scatter plot
plt.scatter(new1["votes"],new1["rate"])
plt.ylabel('Rating')
plt.xlabel('Votes')
plt.show()


#Bar graph for count of restaurants in every locality
	# Make a fake dataset
n=new1["listed_in(city)"].value_counts()
print(sum(n))

height =list(n)
bars = list(n.keys())
y_pos = np.arange(len(bars))
plt.bar(y_pos, height,color=sns.color_palette())
plt.xticks(y_pos, bars,rotation="vertical")
plt.ylim(0,3000)
plt.xlabel("Location")
plt.ylabel("No. of restaurants")
plt.show()


#Bar graph for count of restaurants in every locality
	# Make a fake dataset
n=new1["listed_in(type)"].value_counts()
print(sum(n))

height =list(n)
bars = list(n.keys())
y_pos = np.arange(len(bars))
plt.bar(y_pos, height,color=sns.color_palette())
plt.xticks(y_pos, bars,rotation="vertical")
plt.ylim(0,25000)
plt.xlabel("Type")
plt.ylabel("No. of restaurants")
plt.show()


#Distributions
sum1=0
for i in new1["approx_cost(for two people)"]:
    if i==500:
        sum1+=1


    #For approximate cost
sns.distplot(new1["approx_cost(for two people)"],kde=False)
	#For rating
sns.distplot(new1["rate"],kde=False)



#Pie chart
	#Online ordering
sizes = list(new1["online_order"].value_counts())
labels =["Yes","No"]
 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  

plt.title("Online order")
plt.show()

	#Book table
sizes = list(new1["book_table"].value_counts())
labels =["No","Yes"]
 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  

plt.title("Book table")
plt.show()


