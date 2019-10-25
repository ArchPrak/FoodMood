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


# bar plot ##########
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






