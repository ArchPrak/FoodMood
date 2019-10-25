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

