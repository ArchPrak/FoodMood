import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

#Reading dataset
df=read_csv("../input/zomato-bangalore-restaurants/zomato.csv")


print(type(df))
print(len(df))

#Dropping unnecessary columns
new=df.drop(["url","phone","dish_liked","menu_item"],axis=1)
#print(df.columns)
#print(new.columns)

#Omitting na values
new[new=='']= np.nan
new[new=='NEW']= np.nan
new[new=='-']= np.nan
new1 = new.dropna()

print(len(new1))

#Removing /5 in rating
new1['rate']=new1['rate'].replace("/5","", regex=True)
new1['rate']=new1['rate'].astype('float32')

#Converting Yes and No to 1 and 0
new1.book_table[new1.book_table=="Yes"]=1
new1.book_table[new1.book_table=="No"]=0

new1.online_order[new1.online_order=="Yes"]=1
new1.online_order[new1.online_order=="No"]=0

#Removing unusual characters
new1.name= new1.name.map(lambda x: x.encode('ascii','ignore').decode('ascii'))

#Removing comma and converting datatype to float
new1['approx_cost(for two people)']=new1['approx_cost(for two people)'].replace(",","", regex=True)
new1['approx_cost(for two people)']=new1['approx_cost(for two people)'].astype('float32')

###########################################################################################

#EDA

# scatter plot #############################

import matplotlib.pyplot as plt
plt.scatter(new1["votes"],new1["rate"])
plt.ylabel('Rating')
plt.xlabel('Votes')
plt.title("Scatter plot for Rating vs Votes")
plt.show()


# bar plot ###############################
import seaborn as sns


n=new1["listed_in(city)"].value_counts()
print(sum(n))

height =list(n)
bars = list(n.keys())
y_pos = np.arange(len(bars))
#s=sns.color_palette("husl", 8)
plt.bar(y_pos, height,color=sns.color_palette())
plt.xticks(y_pos, bars,rotation="vertical")
plt.ylim(0,3000)
plt.xlabel("Location")
plt.ylabel("No. of restaurants")
plt.title("Areawise Restaurant count")
plt.show()

# wordcloud ##############################

# Libraries
from wordcloud import WordCloud

data=df

#finding most popular location
#pop_location="Banashankari"
pop_location=data.mode()["location"]

#filtering to get rows with that location
d=data.loc[data["location"] == pop_location[0]]

l=[]
for i in list(d["dish_liked"]):
    if type(i)==str:
        l.extend(i.split(","))

text=" ".join(l) 
# Create the wordcloud object
wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)
 
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("Famous dishes in BTM (area with max food outlets)")
plt.show()


# correlogram ##############################
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(new1.corr(), xticklabels=new1.corr().columns, yticklabels=new1.corr().columns, cmap='RdYlGn', center=0, annot=True)

# Decorations
plt.title('Correlogram ', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()



# hist ##################
plot1=sns.distplot(new1["approx_cost(for two people)"],kde=False)
plot1.set_title("Distribution of Approx cost for 2 people")
plt.show()

plot2=sns.distplot(new1["rate"],kde=False)
plot2.set_title("Distribution of Rating")
plt.show()



# pie chart ################
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
sizes = list(new1["online_order"].value_counts())
labels =["Yes","No"]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Online order")
plt.show()


sizes = list(new1["book_table"].value_counts())
labels =["No","Yes"]
 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Book table")
plt.show()


###LASSO ###############################
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data=new1
Y=new1["rate"]
data=data.drop(["rate","address", "name","location","rest_type","cuisines","reviews_list","listed_in(city)"],axis=1)
X=pd.get_dummies(data["listed_in(type)"],prefix=['Type'])
#X 
result = pd.concat([data, X], axis=1).reindex(data.index)
#result
X=result
X=X.drop(["listed_in(type)"],axis=1)
#X


X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=31)
lasso = Lasso()
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)
print ("training score:", train_score )
print ("test score: ", test_score)
print ("number of features used: ", coeff_used)

lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,y_train)
train_score001=lasso001.score(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)
print ("training score for alpha=0.01:", train_score001 )
print ("test score for alpha =0.01: ", test_score001)
print ("number of features used: for alpha =0.01:", coeff_used001)

lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(X_train,y_train)
train_score00001=lasso00001.score(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)

print ("training score for alpha=0.0001:", train_score00001 )
print ("test score for alpha =0.0001: ", test_score00001)
print ("number of features used: for alpha =0.0001:", coeff_used00001)












