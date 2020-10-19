#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Oct 15 21:24:28 2020

@author: huaweisun
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB
import random
#get data and converting data
def get_age(val):
    if val < 21 or val > 50:
        raise ValueError("you value must bettwen 21 and 50 ")
    elif val <= 25:
        return '21..25'
    elif val <= 30:
        return '26..30'
    elif val <= 35:
        return '31..35'
    elif val <= 40:
        return '36..40'
    elif val <= 45:
        return '41..45'
    else:
        return '46..50'
#get data and converting data
def get_salary(val):
    if val < 26 or (val > 50 and val < 66) or val > 70:
        raise ValueError("you value must bettwen 26 and 50 or 66 and 70")
    elif val <= 30:
        return '26k..30k'
    elif val <= 35:
        return '31k..35k'
    elif val <= 40:
        return '36k..40k'
    elif val <= 45:
        return '41k..45k'
    elif val <= 50:
        return '46k..50k'
    else:
        return '66k..70k'
#this is question 1a. 
print("_____________________Begining 1a___________________________\n")
df1 = pd.read_csv("hwk06-01.csv", delimiter="\t" )    
print(df1)


#this is question 1b
print("_____________________Begining 1b___________________________\n")
df2 = pd.DataFrame(columns=['department','status','age','salary'])
len_count = len(df1['count'])
count =0
for i in range(len_count):
    for j in range(df1['count'][i]):
        # data = pd.DataFrame({'department':df1['department'][i],'status':df1['status'][i], 'age':df1['age'][i], 'salary':df1['salary'][i]}, index =[i])
        # print(data)
        # # concat = pd.concat([data])
        # df2.append(data)
       
        df2.loc[j+count] = [df1['department'][i]] + [df1['status'][i]]+ [df1['age'][i]] + [df1['salary'][i]]
    count +=df1['count'][i]
print(df2)

print("_____________________Begining 1c and d_________\n")
#make encode 0 1 2 3 4 5
dep_Encode = [0,1,2,3,4,5]
age_Encode = [0,1,2,3,4,5]
salary_Encode = [0,1,2,3,4]
count = 0
#for loop through data in df2
for i in range(len(df2)):
    if(df2.loc[i, "department"] == "sales"):
        df2.loc[i, "department"] = dep_Encode[0]

    if(df2.loc[i, "department"] == "systems"):
        df2.loc[i, "department"] = dep_Encode[1]

    if(df2.loc[i, "department"] == "marketing"):
        df2.loc[i, "department"] = dep_Encode[2]

    if(df2.loc[i, "department"] == "secretaty"):
        df2.loc[i, "department"] = dep_Encode[3]

    ##encode age
    if(df2.loc[i, "age"] == "31..35"):
        df2.loc[i, "age"] = age_Encode[0]

    if(df2.loc[i, "age"] == "26..30"):
        df2.loc[i, "age"] = age_Encode[1]

    if(df2.loc[i, "age"] == "21..25"):
        df2.loc[i, "age"] = age_Encode[2]

    if(df2.loc[i, "age"] == "41..45"):
        df2.loc[i, "age"] = age_Encode[3]

    if(df2.loc[i, "age"] == "36..40"):
        df2.loc[i, "age"] = age_Encode[4]

    if(df2.loc[i, "age"] == "46..50"):
        df2.loc[i, "age"] = age_Encode[5]

    #Encode salary
    if(df2.loc[i, "salary"] == "26k..30k"):
        df2.loc[i, "salary"] = salary_Encode[0]

    if(df2.loc[i, "salary"] == "31k..35k"):
        df2.loc[i, "salary"] = salary_Encode[1]

    if(df2.loc[i, "salary"] == "36k..40k"):
        df2.loc[i, "salary"] = salary_Encode[1]


    if(df2.loc[i, "salary"] == "41k..45k"):
        df2.loc[i, "salary"] = salary_Encode[2]

    if(df2.loc[i, "salary"] == "46k..50k"):
        df2.loc[i, "salary"] = salary_Encode[3]

    if(df2.loc[i, "salary"] == "66k..70k"):
        df2.loc[i, "salary"] = salary_Encode[4]
alist =[]
df2_list = df2.values.tolist()
for i in range (len(df2_list)):
    alist.append(df2_list[i])
#make a list and with index
for i in range(len(alist)):
    print(i, alist[i])



#this is question 2
print("_____________________Begining 2__________________________\n")
# pipeline to vectorize dict input to Decision Tree classifier
pipe = make_pipeline(DictVectorizer(), DecisionTreeClassifier())

# data for training model 
Train_Xdata = df2[['department', 'age', 'salary']].to_dict('records')
Train_Ydata = df2['status']


# starting fit to training data
pipe.fit(Train_Xdata, Train_Ydata)

# predict the status of a user provided unseen data, such as
#t =< department : systems, status :?, age : 28, salary : 50K >
user_str = ('system ' + '28 ' + '50k')


# split user data
query = user_str.split(' ')
age = get_age(int(query[1]))
salary = get_salary(int(query[2].rstrip('k')))
tup_dict = {'department': query[0], 'age': age, 'salary': salary}


# get prediction from user data
pred = pipe.predict(tup_dict)

# print predicted by model
print('display decision tree prediction of {}:'.format(tup_dict))
print(pred[0])



#this is question 3
print("_____________________Begining 3a_____________________\n")
df3 = pd.DataFrame(columns=['department','status','age','salary'])

#get the len of df1
len_count = len(df1['count'])
count =0
#make copy of df1 with number of copies as indicated in the
#count column
for i in range(len_count):
    for j in range(df1['count'][i]):
        df3.loc[j+count] = [df1['department'][i]] + [df1['status'][i]]+ [df1['age'][i]] + [df1['salary'][i]]
    count +=df1['count'][i]
    
    
print("_____________________Begining 3b and c_______________\n")
#encord
dep_Encord = [0,1,2,3,4,5]
count = 0
#print data in df3 and make randomlizing data
for i in range(len(df3)):
    if(df3.loc[i, "department"] == "sales"):
        df3.loc[i, "department"] = dep_Encord[0]

    if(df3.loc[i, "department"] == "systems"):
        df3.loc[i, "department"] = dep_Encord[1]

    if(df3.loc[i, "department"] == "marketing"):
        df3.loc[i, "department"] = dep_Encord[2]

    if(df3.loc[i, "department"] == "secretaty"):
        df3.loc[i, "department"] = dep_Encord[3]

    #Encode age
    if(df3.loc[i, "age"] == "31..35"):
        df3.loc[i, "age"] = random.randint(31, 36)

    if(df3.loc[i, "age"] == "26..30"):
        df3.loc[i, "age"] = random.randint(26, 31)

    if(df3.loc[i, "age"] == "21..25"):
        df3.loc[i, "age"] = random.randint(21, 26)

    if(df3.loc[i, "age"] == "41..45"):
        df3.loc[i, "age"] = random.randint(14, 46)

    if(df3.loc[i, "age"] == "36..40"):
        df3.at[i, "age"] = random.randint(36, 40) 

    if(df3.loc[i, "age"] == "46..50"):
        df3.loc[i, "age"] = random.randint(46, 51)

    ##Encode salary

    if(df3.loc[i, "salary"] == "26k..30k"):
        df3.loc[i, "salary"] = str(random.randint(26, 31)) +'K'

    if(df3.loc[i, "salary"] == "31k..35k"):
        df3.loc[i, "salary"] =str(random.randint(31, 36)) +'K'

    if(df3.loc[i, "salary"] == "36k..40k"):
        df3.loc[i, "salary"] = str(random.randint(36, 41)) +'K'


    if(df3.loc[i, "salary"] == "41k..45k"):
        df3.loc[i, "salary"] = str(random.randint(41, 46)) +'K'

    if(df3.loc[i, "salary"] == "46k..50k"):
        df3.loc[i, "salary"] = str(random.randint(46, 51)) +'K'

    if(df3.loc[i, "salary"] == "66k..70k"):
        df3.loc[i, "salary"] =str(random.randint(66, 71)) +'K'
#Print out df3 after randomlizing
print(df3)

print("_____________________Begining 4_______________\n")


# vectorize dict input and Multinomial Naive Bayes classifier
pipe = make_pipeline(DictVectorizer(), MultinomialNB())

#data from training model
train_X = df3[['department', 'age', 'salary']].to_dict('records')
train_y = df3['status']

# fit to traning data
pipe.fit(train_X, train_y)

# predictive model to predict the status of a
#user provided unseen data
#t =< department : systems, status :?, age : 28, salary : 50K >
user_str = ('system ' + '28 ' + '50k')

# split user data
User_data = user_str.split(' ')
tup_dict = {'department': User_data[0], 'age': User_data[1], 'salary': User_data[2]}

# prediction data
pred = pipe.predict(tup_dict)

# print out Naive Bays prediction
print('display: Naive Bayes Prediction of {}:'.format(tup_dict))
print(pred[0])




