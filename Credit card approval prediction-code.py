#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[2]:


app = pd.read_csv('application_record.csv')
credit = pd.read_csv('credit_record.csv')


# In[3]:


app.head()


# In[4]:


app.tail()


# In[5]:


credit.head()


# In[6]:


credit.tail()


# In[7]:


print("Number of people working status:")
print(app['OCCUPATION_TYPE'].value_counts())
sns.set(rc ={'figure.figsize':(18,6)})
sns.countplot(x='OCCUPATION_TYPE',data=app,palette = 'Set2')


# In[8]:


print('Types of house of the people:')
print(app['NAME_HOUSING_TYPE'].value_counts())
sns.set(rc = {'figure.figsize':(15,4)})
sns.countplot(x='NAME_HOUSING_TYPE',data=app,palette ='Set2')


# In[9]:


print("Income types of the Person:")
print(app['NAME_INCOME_TYPE'].value_counts())
sns.set(rc ={'figure.figsize':(8,5)})
sns.countplot(x='NAME_INCOME_TYPE',data=app,palette ='Set2')


# In[10]:


flg, ax=plt.subplots(figsize=(8,6))
sns.heatmap(app.corr(),annot=True)


# In[11]:


app.describe()


# In[12]:


app.info()


# In[13]:


def unique_values():
    a = app.CODE_GENDER.unique()
    print("-------CODE_GENDER---------")
    print(a)
    print()
    b = app.FLAG_OWN_CAR.unique()
    print("--------FLAG_OWN_CAR---------")
    print(b)
    print()
    c = app.FLAG_OWN_REALTY.unique()
    print("-------FLAG_OWN_REALTY---------")
    print(c)
    print()
    d = app.CNT_CHILDREN.unique()
    print("---------CNT_CHILDREN------")
    print(d)
    print()
    e = app.NAME_INCOME_TYPE.unique()
    print("---------NAME_INCOME_TYPE------")
    print(e)
    print()
    f = app.NAME_EDUCATION_TYPE.unique()
    print("-------NAME_EDUCATION_TYPE--------")
    print(f)
    print()
    g = app.NAME_FAMILY_STATUS.unique()
    print("------NAME_FAMILY_STATUS------")
    print(g)
    print()
    h = app.NAME_HOUSING_TYPE.unique()
    print("------NAME_HOUSING_TYPE------")
    print(h)
    print()
    i = app.OCCUPATION_TYPE.unique()
    print("------OCCUPATION_TYPE-----")
    print(i)
    print()
    j = app.CNT_FAM_MEMBERS.value_unique()
    print("-------CNT_FAM_MEMBERS------")
    print(j)
    print()
    return unique_values


# In[14]:


app.drop_duplicates(subset = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','DAYS_BIRTH','DAYS_EMPLOYED','FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL','OCCUPATION_TYPE','CNT_FAM_MEMBERS'],keep = 'first',inplace = True)


# In[15]:


app.isnull().mean()


# In[16]:


def data_cleansing(data):
    data['CNT_FAM_MEMBERS'] = data['CNT_FAM_MEMBERS'] + data['CNT_CHILDREN']
    dropped_cols = ['FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL','OCCUPATION_TYPE','CNT_CHILDREN']
    data = data.drop(dropped_cols,axis = 1)
    
    data['DAYS_BIRTH'] = np.abs(data['DAYS_BIRTH']/365)
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED']/365
    
    housing_type = {'House / apartment' : 'House / apartment',
                   'with parents' : 'With parents',
                   'Municipal apartment' : 'House / apartment',
                   'Rented apartment' : 'House / apartment',
                   'Office apartment' : 'House / apartment',
                   'co-op apartment' : 'House / apartment'}
    income_type = {'Commercial associate' : 'Working',
                  'State servant' : 'Woking',
                  'Working' : 'Working',
                  'Pensioner' : 'Pensioner',
                  'Student' : 'Student'}
    education_type = {'Secondary / secondary special':'secondary',
                     'Lower secondary' : 'Secondary',
                     'Higher education' : 'Higher education',
                     'Incomplete higher' : 'Higher education',
                     'Academic degree' : 'Academic degree'}
    family_status = {'Single / not married' : 'Single',
                    'Separated' : 'Single',
                    'widow' : 'Single',
                    'Civil marriage' : 'Married',
                    'Married' : 'Married'}
    data['NAME_HOUSING_TYPE'] = data['NAME_HOUSING_TYPE'].map(housing_type)
    data['NAME_INCOME_TYPE'] = data['NAME_INCOME_TYPE'].map(income_type)
    data['NAME_EDUCATION_TYPE'] = data['NAME_EDUCATION_TYPE'].map(education_type)
    data['NAME_FAMILY_STATUS'] = data['NAME_FAMILY_STATUS'].map(family_status)
    return data


# In[17]:


credit.head()


# In[18]:


credit.shape


# In[19]:


credit.info


# In[20]:


credit.info()


# In[21]:


#data frame to analyze lenght of the time since intial approval of the credit card
#show number of past dues, paid off and o loan status
grouped = credit.groupby('ID')

pivot_tb = credit.pivot(index = 'ID',columns = 'MONTHS_BALANCE',values = 'STATUS')
pivot_tb['open_month'] = grouped['MONTHS_BALANCE'].min()
pivot_tb['end_month'] = grouped['MONTHS_BALANCE'].max()
pivot_tb['window'] = pivot_tb['end_month'] - pivot_tb['open_month']
pivot_tb['window'] += 1

pivot_tb['paid_off'] = pivot_tb[pivot_tb.iloc[:,0:61] == 'C'].count(axis = 1)
pivot_tb['pastdue_1-29'] = pivot_tb[pivot_tb.iloc[:,0:61] == '0'].count(axis = 1)
pivot_tb['pastdue_30-59'] = pivot_tb[pivot_tb.iloc[:,0:61] == '1'].count(axis = 1)
pivot_tb['pastdue_60-89'] = pivot_tb[pivot_tb.iloc[:,0:61] == '2'].count(axis = 1)
pivot_tb['pastdue_90-119'] = pivot_tb[pivot_tb.iloc[:,0:61] == '3'].count(axis = 1)
pivot_tb['pastdue_120-149'] = pivot_tb[pivot_tb.iloc[:,0:61] == '4'].count(axis = 1)
pivot_tb['pastdue_over_150'] = pivot_tb[pivot_tb.iloc[:,0:61] == '5'].count(axis = 1)
pivot_tb['no_loan'] = pivot_tb[pivot_tb.iloc[:,0:61] == 'X'].count(axis = 1)

# setting Id column to merge with app data.
pivot_tb['ID'] = pivot_tb.index


# In[22]:


pivot_tb.head(15)


# In[23]:


pivot_tb.tail(15)


# In[24]:


def feature_engineering_target(data):
    good_or_bad = []
    for index, row in data.iterrows():
        paid_off = row['paid_off']
        over_1 = row['pastdue_1-29']
        over_30 = row['pastdue_30-59']
        over_60 = row['pastdue_60-89']
        over_90 = row['pastdue_90-119']
        over_120 = row['pastdue_120-149'] + row['pastdue_over_150']
        no_loan = row['no_loan']
        
        overall_pastdues = over_1+over_30+over_60+over_90+over_120
        if overall_pastdues == 0:
            if paid_off >=no_loan or paid_off <= no_loan:
                good_or_bad.append(1)
            elif paid_off == 0 and no_loan == 1:
                good_or_bad.append(1)
        elif overall_pastdues !=0:
            if paid_off > overall_pastdues:
                good_or_bad.append(1)
            elif paid_off <= overall_pastdues:
                good_or_bad.append(0)
        elif paid_off == 0 and no_loan != 0:
            if overall_pastdues <= no_loan or overall_pastdues >= no_loan:
                good_or_bad.append(0)
        else:
            good_or_bad.append(1)
            
    return good_or_bad 


# In[25]:


target = pd.DataFrame()
target['ID'] = pivot_tb.index
target['paid_off'] = pivot_tb['paid_off'].values
target['#_of_pastdues'] = pivot_tb['pastdue_1-29'].values + pivot_tb['pastdue_30-59'].values 
+ pivot_tb['pastdue_60-89'].values + pivot_tb ['pastdue_90-119'].values 
+ pivot_tb['pastdue_120-149'].values + pivot_tb['pastdue_over_150'].values

target['no_loan'] = pivot_tb['no_loan'].values
target['target'] = feature_engineering_target(pivot_tb)
credit_app = app.merge(target, how ='inner' ,on = 'ID')
credit_app.drop('ID',axis = 1,inplace = True)


# In[26]:


credit_app


# In[27]:


from sklearn.preprocessing import LabelEncoder

cg=LabelEncoder ()

oc=LabelEncoder ()

own_r= LabelEncoder ()

it = LabelEncoder ()

et = LabelEncoder ()

fs= LabelEncoder ()

ht = LabelEncoder ()

nf = LabelEncoder ()

nh = LabelEncoder ()

o =  LabelEncoder ()

credit_app['CODE_GENDER'] = cg.fit_transform(credit_app[ 'CODE_GENDER']) 
credit_app['FLAG_OWN_CAR'] = oc.fit_transform(credit_app['FLAG_OWN_CAR'])
credit_app['FLAG_OWN_REALTY']= own_r.fit_transform(credit_app[ 'FLAG_OWN_REALTY']) 
credit_app['NAME_INCOME_TYPE'] = it.fit_transform (credit_app['NAME_INCOME_TYPE']) 
credit_app['NAME_EDUCATION_TYPE'] = et.fit_transform(credit_app['NAME_EDUCATION_TYPE']) 
credit_app['NAME FAMILY STATUS'] = fs.fit_transform(credit_app['NAME_FAMILY_STATUS']) 
credit_app['NAME HOUSING_TYPE'] = ht.fit_transform(credit_app['NAME_HOUSING_TYPE'])
credit_app['NAME_FAMILY_STATUS'] = nf.fit_transform(credit_app['NAME_FAMILY_STATUS'])
credit_app['NAME_HOUSING_TYPE'] = nf.fit_transform(credit_app['NAME_HOUSING_TYPE'])
credit_app['OCCUPATION_TYPE'] = nf.fit_transform(credit_app['OCCUPATION_TYPE'])


# In[28]:


x = credit_app[credit_app.drop('target', axis = 1).columns]
y = credit_app['target']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size = 0.8, random_state = 0)


# In[29]:


x.info()


# In[30]:


def logistic_reg(xtrain,xtest, ytrain, ytest): 
  Ir-LogisticRegression (solver-'liblinear')
  lr.fit(xtrain, ytrain)
  ypred-lr.predict(xtest) 
  print('***LogisticRegression***')
  print('Confusion matrix') 
  print(confusion_matrix(ytest,ypred))
  print('Classification report')
  print(classification_report(ytest, ypred))


# In[31]:


def random_forest(xtrain, xtest, ytrain,ytest): 
  rf=RandomForestClassifier()
  rf.fit(xtrain, ytrain)
  ypred-rf.predict(xtest) 
  print('***RandomForestClassifier***')
  print('Confusion matrix') 
  print(confusion_matrix(ytest,ypred)) 
  print('Classification report')
  print(classification_report(ytest,ypred))


# In[32]:


def g_boosting (xtrain, xtest, ytrain,ytest):
   gb=GradientBoostingClassifier()
   gb.fit(xtrain, ytrain) 
   ypred=gb.predict(xtest)
   print('***Gradient BoostingClassifier***') 
   print('Confusion matrix') 
   print(confusion_matrix(ytest,ypred)) 
   print('Classification report') 
   print(classification_report(ytest, ypred))


# In[33]:


def d_tree(xtrain,xtest, ytrain,ytest):
   dt=DecisionTreeClassifier() 
   dt.fit(xtrain, ytrain) 
   ypred=dt.predict(xtest) 
   print('***DecisionTreeClassifier***') 
   print('Confusion matrix') 
   print(confusion_mattix(ytest,ypred)) 
   print('Classification report') 
   print(classification_report(ytest,ypred))


# In[34]:


def compare_model (xtrain,xtest, ytrain,ytest): 
  logistic_reg(xtrain, xtest, ytrain,ytest)
  print('-'*100)
  random_forest(xtrain, xtest, ytrain,ytest)
  print('-'*100)
  g_boosting(xtrain, xtest, ytrain,ytest)
  print('-'*100)
  d_tree(xtrain, xtest, ytrain,ytest) 


# In[35]:


dt = DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
ypred = dt.predict(xtest)


# In[37]:


import pickle
pickle.dump(dt,open("model.pk1","wb"))


# In[ ]:




