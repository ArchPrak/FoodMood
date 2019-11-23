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



######################## Lasso and Linear Regression ####################################

data=new1
Y=new1["rate"]
data=data.drop(["rate","address", "name","location","rest_type","cuisines","reviews_list","listed_in(city)"],axis=1)
X=pd.get_dummies(data["listed_in(type)"])
result = pd.concat([data, X], axis=1).reindex(data.index)
X=result
X=X.drop(["listed_in(type)"],axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=31)

lasso = Lasso()
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)
print ("training score:", train_score )
print ("test score: ", test_score)
print ("number of features used: ", coeff_used)

#Alpha= 0.01
lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,y_train)
train_score001=lasso001.score(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)
print ("training score for alpha=0.01:", train_score001 )
print ("test score for alpha =0.01: ", test_score001)
print ("number of features used: for alpha =0.01:", coeff_used001)

#Alpha= 0.0001
lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(X_train,y_train)
train_score00001=lasso00001.score(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)
print ("training score for alpha=0.01:", train_score00001 )
print ("test score for alpha =0.01: ", test_score00001)
print ("number of features used: for alpha =0.01:", coeff_used00001)

#Linear regression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_train_score=lr.score(X_train,y_train)
lr_test_score=lr.score(X_test,y_test)
print ("LR training score:", lr_train_score )
print ("LR test score: ", lr_test_score)


########################### Feature Selection #################################

new1.groupby(['listed_in(city)']) #Grouping dataframe by locality

#Getting counts of restaurants in each locality
dloc={}

for i in range(len(new1)):
    if(new1["listed_in(city)"][i] not in dloc):
        dloc[new1["listed_in(city)"][i]]=0
    dloc[new1["listed_in(city)"][i]]+=1


#Getting required indices in dataframe for each locality
i=0
start=0
end=0
locind=[]
for i in dloc:
    end=start+dloc[i]-1
    locind.append((start,end))
    start=end+1

print(len(locind))


#Decision tree function
def dt(df, rate, location):
	#Features used in the decision tree
    feat_labels=["online_order", "book_table", "votes", "approx_cost(for two people)", "listed_in(type)"]

    data=df

    #Getting one-hot encoding of restaurant type
    X= pd.get_dummies(data,columns=['listed_in(type)'])

    y=[]

    #df is a dataframe and each row is a tuple with all features at index [1]
    
    for i in rate:
        if(i>=3.8):
            y.append(1)
        else:
            y.append(0)
        
    # Split the data into 40% test and 60% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    # Create a random forest classifier
    clf = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=10, min_samples_leaf=5)

    # Train the classifier
    clf.fit(X_train, y_train)
    
    y_pred=clf.predict(X_test)
    a=accuracy_score(y_test,y_pred)

    out=[]
    # Appending the name and gini importance of each feature
    for feature in zip(feat_labels, clf.feature_importances_):
        out.append(feature)

    print("%s: %f"%(location,a*100),"%")
    return out



#Function to call the decision tree for every locality
def makedf(i,j,orgdf):
    locdf=orgdf[i:j+1]
    location=list(locdf["listed_in(city)"])[1]
    rate=list(locdf["rate"])
    locdf=locdf.drop(["address","name","rate","location","rest_type","cuisines","reviews_list","listed_in(city)"],axis=1)
    l=dt(locdf,rate, location)
    return (location,l)


print("The accuracies of the decision tree prediction for various locations are as follows:")
locfeatures=[]
for i in range(len(locind)):
    locfeatures.append(makedf(locind[i][0],locind[i][1],new1))



print("The top 3 features influencing the food culture of the locations are:")
for i in range(len(locfeatures)):
    print(locfeatures[i][0],':')
    locfeatures[i][1].sort(key=lambda x: x[1], reverse=True)
    print("   ",locfeatures[i][1][0][0],'-->',round(locfeatures[i][1][0][1]*100, 2),"%")
    print("   ",locfeatures[i][1][1][0],'-->',round(locfeatures[i][1][1][1]*100, 2),"%")
    print("   ",locfeatures[i][1][2][0],'-->',round(locfeatures[i][1][2][1]*100, 2),"%")



############################# Evaluating the cost range #####################################

#Grouping approx. costs into bins of varying sizes
new2=new1.copy(deep=False)
new2['cost_bins'] = pd.cut(x=new1['approx_cost(for two people)'], bins=[0, 300, 600, 1200, 2000, 4000,6000])

cbd= {}
label=0
for i in range(len(new2)):
    if(new2['cost_bins'][i] not in cbd):
        cbd[new2['cost_bins'][i]]=label
        label+=1

l=[]
for i in range(len(new2)):
    l.append(cbd[new2['cost_bins'][i]])
    
new2['cb_class']=l



#Logistic Regression
pred_final=[]
act_final=[]
d=new2.copy(deep=False)

y= list(d['cb_class'])
x_= d.drop(["address","name","rate","location","rest_type","cuisines","approx_cost(for two people)","reviews_list","cost_bins",'cb_class'],axis=1)

	#Getting one-got encoding of restaurant type and location
x= pd.get_dummies(x_,columns=['listed_in(type)','listed_in(city)'])


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)

	#Normalizing data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

	#Model
model= LogisticRegression(solver = 'lbfgs')
model.fit(X_train,y_train)
pred=model.predict(X_test)
act_final=y_test
pred_final=pred


print("The accuracy of the predicted cost for two people using logistic regression is:")
print(round(accuracy_score(act_final,pred_final)*100, 2),"%\n")

print("The confusion matrix for the model is printed:")
print("   0 --> 600 to 1200 Rs.")
print("   1 --> 0 to 300 Rs.")
print("   2 --> 300 to 600 Rs.")
print("   3 --> 1200 to 2000 Rs.")
print("   4 --> 2000 to 4000 Rs.")
print("   5 --> 4000 to 6000 Rs.\n")
print(confusion_matrix(act_final,pred_final,labels=[0,1,2,3,4,5]))



#Artificial Neural Network


d=new2.copy(deep=False)

y= pd.get_dummies(d['cb_class'])
x_= d.drop(["address","name","rate","location","rest_type","cuisines","approx_cost(for two people)","reviews_list","cost_bins",'cb_class'],axis=1)

	#Getting one-got encoding of restaurant type and location
x= pd.get_dummies(x_,columns=['listed_in(type)','listed_in(city)'])


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)

	#Normalizing data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
    

    #Model
model = Sequential()
model.add(Dense(32, input_dim=40, init='uniform', activation='relu'))
model.add(Dense(6, init='uniform', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, nb_epoch=50, batch_size=50,  verbose=1)

	#Print Accuracy
scores = model.evaluate(X_test, y_test) 

pred= model.predict(X_test)


l=y_test.values.tolist()
act_final=[l[i].index(max(l[i])) for i in range(len(l))]
m=list(pred)
pred_final=[list(m[i]).index(max(m[i])) for i in range(len(m))]

print("The accuracy of the predicted cost for two people using neural network is:")
print(round(scores[1]*100,2),"%\n")

print("The confusion matrix for the model is printed:")
print("   0 --> 600 to 1200 Rs.")
print("   1 --> 0 to 300 Rs.")
print("   2 --> 300 to 600 Rs.")
print("   3 --> 1200 to 2000 Rs.")
print("   4 --> 2000 to 4000 Rs.")
print("   5 --> 4000 to 6000 Rs.\n")
print(confusion_matrix(act_final,pred_final,labels=[0,1,2,3,4,5]))


######################### Location wise bar plot ######################################

#Function for plotting location wise bar plot
def myplot(feat,i,j,df):
    df=df[i:j+1]
    df=df[df["rate"]>3.8]
    n=df[feat].value_counts()
    l=list(zip(list(n),list(n.keys())))
    l.sort(key=lambda tup:tup[1])
    height =[x[0] for x in l]
    bars = [x[1] for x in l]

    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height,color=sns.color_palette())
    plt.xticks(y_pos, bars,rotation="vertical")
    plt.xlabel("Suitable values")
    plt.ylabel("Count")
    plt.title("Values for "+ feat)
    plt.show()
    
    
#Dictionary : {location : index in locfeature list}
d={}
for i in range(len(locfeatures)):
    d[locfeatures[i][0]]=i  

#Sort in desc order of importance
for location in d:
    locfeatures[d[location]][1].sort(key=lambda tup: tup[1],reverse =True)

#Location
l="Banashankari"

#2 top features
f1=locfeatures[d[l]][1][0][0]
f2=locfeatures[d[l]][1][1][0]

#Start and end for that location
start,end=locind[d[l]]

#Make subset and plot 
myplot(f1,start,end,new1)
myplot(f2,start,end,new1)




################################## Sentiment scores #######################################
### !!!!! This code takes a lot of time to run and hence the results are stored in a file pos_list.txt !!!!!

#Sentiment scores analyzer
analyzer=SentimentIntensityAnalyzer()
english_stop_words = stopwords.words('english')

def remove_stop_words(corpus):
    l=[]
    words=corpus.split()
    for word in words:
        if word not in english_stop_words:
            l.append(word)
    return " ".join(l)


def sentscore(df): 
    pos_list=[]

    for i in range(len(df)):
        line= list(df["reviews_list"])[i]
    
    	#Stop word removal for every row and other processing
        review = re.sub('\\n',"", line.lower()) 
        review= re.sub('[^a-zA-Z]',' ',review.lower())
        review = re.sub('rated',"", review.lower()) 
        review = remove_stop_words(review)
    
    	#Getting the positive scores
        pos=(analyzer.polarity_scores(review))['pos']
    
        pos_list.append(pos)
    
        print(i,pos)
        
    return pos_list


sentscore(new1)


############################## Feature value prediction ###################################


#Decision tree function
def dt(df, rate, testx):
    feat_labels=["online_order", "book_table", "cb_class", "listed_in(type)"]

    data=df

    l=pd.Series(LabelEncoder().fit_transform(data['listed_in(type)']))
    
    X= pd.concat([data, l], axis=1).reindex(data.index)
    X=X.drop(['listed_in(type)'],axis=1)

    #Label encoding the restaurant types
    l1=pd.Series(LabelEncoder().fit_transform(testx['listed_in(type)']))
    testx= pd.concat([testx, l1], axis=1).reindex(testx.index)
    
    
    test=testx.drop(['listed_in(type)'],axis=1)
    y=list()

    for i in rate:
        j= round(i,0)
        y.append(str(j))

    # Split the data into 40% test and 60% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

    # Train the classifier
    clf.fit(X_train, y_train)
    
    y_pred=clf.predict(X_test)
    
    y_pred_new = clf.predict(test)
    a=accuracy_score(y_test,y_pred)

    print(a)
    
    testx['pred_ratings']=y_pred_new
    return testx



def bestvalues(l,i,j,df):
    df=df[i:j+1]
    df.index=(range(0,len(df)))
    rate= df['rate']
    df=df.drop(["rate"],axis=1)
    
    #Set of values for all columns
    u1=set(list(df["online_order"]))
    u2=set(list(df["book_table"]))
    u4=set(list(df["listed_in(type)"]))
    u3=set(list(df["cb_class"]))
    
    testx=[]
    
    for i in u1:
        for j in u2:
            for k in u3:
                for l in u4:
                    testx.append([i,j,k,l])

    #Model trained with df 
    #testx predicted
    #Max value of rating/sentscore fiund, corresponding row returned 
    
    testx=pd.DataFrame(testx,columns=["online_order","book_table","cb_class","listed_in(type)"])
    return dt(df,rate,testx)

#Some location
l="Whitefield"    


d={}
locations=list(dloc.keys())
for i in range(len(locations)):
    d[locations[i]]=i 


start, end= locind[d[l]]
df=new2[["online_order","rate","book_table","cb_class","listed_in(type)"]]

#Function called for location l
best=bestvalues(l,start,end,df)

#Max rating found and all rows with that rating are chosen
big= max(best['pred_ratings'])
bestest=best[best["pred_ratings"]>=big]
bestest.index = range(len(bestest))


#List out keys and values separately 
key_list = list(cbd.keys()) 
val_list = list(cbd.values()) 

print("Suggested combinations for the upcoming restaurant:\n")
  
print("Online order   |   Book table   |   Cost range   |   Restaurant type")
print("---------------------------------------------------------------------")
for i in range(len(bestest)):
    oo= bestest['online_order'][i]
    bt= bestest['book_table'][i]
    cb= bestest['cb_class'][i]
    typ= bestest['listed_in(type)'][i]
    
    d={0:"No", 1:"Yes"}
    
    print(d[oo],"           |",d[bt],"             |",str(key_list[val_list.index(cb)]).lstrip('(').rstrip(']'),"  |",typ)

