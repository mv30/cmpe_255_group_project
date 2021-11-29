#!/usr/bin/env python
# coding: utf-8

# In[150]:


import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# In[123]:


app_data = pd.read_csv("Google-Playstore.csv")
app_data.head()


# In[66]:


app_data.info()


# In[67]:


original_dim = app_data.shape
print(original_dim)


# In[68]:


app_data["Price"].value_counts()


# In[69]:


app_data['Price'].count(), 2268011/2312944


# Since over 98% of applications are free, the attribute 'Price' will be inconsequential for predicting Ratings.
# Dropping columns like App Id, Currency etc that are not contributing to further analysis.

# In[124]:


app_data.drop(['App Id', 'Currency', 'Developer Id', 'Developer Website', 'Developer Email', 'Privacy Policy', 'Editors Choice', 'Scraped Time','Price'], axis=1, inplace=True)
app_data.head()


# In[71]:


print(app_data.isnull().sum())


# In[72]:


reduced_dim = app_data.shape
print(reduced_dim)


# In[125]:


#Dropping rows with null value for particular columns
app_data.dropna(subset=['Rating', 'Rating Count', 'Installs', 'Size', 'Minimum Android','Released'], inplace=True)
reduced_dim = app_data.shape


# Preprocessing Content Rating
# Mature 17+      => Adult, 
# Adults only 18+ => Adult, 
# Everyone 10+    => Everyone

# In[102]:


app_data['Content Rating'].value_counts()


# In[126]:


app_data['Content Rating'] = app_data['Content Rating'].replace('Unrated',"Everyone")

#Cleaning other values just to include Everyone, Teens and Adult 

app_data['Content Rating'] = app_data['Content Rating'].replace('Mature 17+',"Adults")
app_data['Content Rating'] = app_data['Content Rating'].replace('Adults only 18+',"Adults")
app_data['Content Rating'] = app_data['Content Rating'].replace('Everyone 10+',"Everyone")


# Preprocessing Content Installs
# replaced commmas with no space & dropped + s

# In[127]:


app_data.Installs = app_data.Installs.str.replace(',','')
app_data.Installs = app_data.Installs.str.replace('+','')


# In[128]:


app_data['Installs'] = pd.to_numeric(app_data['Installs'])
app_data['Installs'] = app_data.apply( lambda row: math.log2(row.Installs+1), axis = 1)


# In[78]:


app_data.head()


# Converting app size from string to numeric

# In[129]:


def get_app_size( size_str):
    if size_str == 'Varies with device':
        return 0
    metric_unit = size_str[-1]
    val = float(size_str[:-1])
    if metric_unit == 'k' or metric_unit == 'K':
        val = val*1000.0
    elif metric_unit == 'M':
        val = val*1000000.0
    elif metric_unit == 'G':
        val = val*1000000000.0
    else:
        raise Exception(" Value metric not detected {} ".format(size_str))
    return math.log2(val)


# In[130]:


app_data.Size = app_data.Size.str.replace(',','')
app_data['Size_Log'] = app_data.apply( lambda row: get_app_size(row.Size), axis = 1)


# In[131]:


def processDate(df, columnName):
    # global df
    df_processed =  pd.to_datetime(df[columnName], format='%b %d, %Y')
    df_processed = pd.DataFrame(df_processed)

    df_processed['year'] =  df_processed[columnName].dt.year.astype('Int64')
    df_processed['month'] = df_processed[columnName].dt.month.astype('Int64')

    df['Year '+columnName]=df_processed['year']
    df['Month '+columnName]=df_processed['month']
    df=df.drop([columnName], axis = 1)
    return df


# In[132]:


app_data=processDate(app_data, 'Released')
app_data=processDate(app_data, 'Last Updated')


# In[83]:


app_data.head()


# In[84]:


Var_Corr = app_data.corr()
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns)


# In[85]:


def get_version_value( version):
    if version == 'up':
        return 11
    if version[-1] == 'W':
        version = version[:-1]
    version = version.split('.')
    version.append('0')
    return float("{}.{}".format(version[0], version[1]))


# In[86]:


def get_min_version( version):
    if version == 'Varies with device':
        return 0
    elif '-' in version:
        arr = version.split('-')
        return get_version_value(arr[0].strip())
    elif 'and' in version:
        arr = version.split('and')
        return get_version_value(arr[0].strip())
    return get_version_value( version)


# In[87]:


def get_max_version( version):
    if version == 'Varies with device':
        return 11
    elif '-' in version:
        arr = version.split('-')
        return get_version_value(arr[1].strip())
    elif 'and' in version:
        arr = version.split('and')
        return get_version_value(arr[1].strip())
    return get_version_value( version)


# In[133]:


app_data['Min_Version'] = app_data.apply( lambda row: get_min_version(row['Minimum Android']), axis = 1)
app_data['Max_Version'] = app_data.apply( lambda row: get_max_version(row['Minimum Android']), axis = 1)


# In[134]:


app_data.head()


# In[135]:


app_data.drop(['Minimum Installs', 'Maximum Installs','Minimum Android'], axis=1, inplace=True)


# In[136]:


app_data.head()


# In[137]:


category_install = app_data.groupby(['Category'])['Installs'].agg('sum')
category_install


# In[122]:


app_data.head()


# In[138]:


plt.figure(figsize=(20,15))
sns.barplot(category_install.index, category_install.values)
plt.title('Number of Installs Per Category')
plt.xlabel('Category')
plt.ylabel('Installs')
plt.xticks(rotation=45,ha='right');


# In[139]:


category_rating = app_data.groupby(['Category'])['Rating'].agg('mean')
plt.figure(figsize=(20,15))
sns.barplot(category_rating.index, category_rating.values)
plt.title('Ratings based on Category')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.xticks(rotation=45,ha='right');


# In[140]:


time_install = app_data.groupby(['Month Released'])['Installs'].agg('sum')
plt.figure(figsize=(15,15))
sns.barplot(time_install.index, time_install.values)
plt.title('Installs based on the month released')
plt.xlabel('Month Released')
plt.ylabel('Installs')


# In[141]:


CR_install = app_data.groupby(['Content Rating'])['Rating'].agg('mean')
plt.figure(figsize=(10,15))
sns.barplot(CR_install.index, CR_install.values)
plt.title('Installs based on the Content Type')
plt.xlabel('Content Rating')
plt.ylabel('Installs')


# In[35]:


sns.lineplot(data=app_data,x='Installs',y='Rating').set(title='Co-relation between Install count and Rating of an App')


# Encode some of the columns as an enumerated type or categorical variable.

# In[142]:


app_data['Category'] = pd.factorize(app_data['Category'])[0].astype(int)
app_data['Content Rating'] = pd.factorize(app_data['Content Rating'])[0].astype(int)
app_data['Ad Supported'] = pd.factorize(app_data['Ad Supported'])[0].astype(int)
app_data['In App Purchases'] = pd.factorize(app_data['In App Purchases'])[0].astype(int)
app_data['Free'] = pd.factorize(app_data['Free'])[0].astype(int)
app_data.head()


# In[143]:


y = app_data['Rating'].values.round()
print(y[0:10])


# In[144]:


X = app_data.drop(['App Name', 'Rating', 'Rating Count', 'Size', 'Year Released', 'Month Released', 'Year Last Updated','Month Last Updated'], axis=1)
X.head()


# In[145]:


std_scaler = StandardScaler()
X['Installs'] = std_scaler.fit_transform(X[['Installs']])
X['Size_Log'] = std_scaler.fit_transform(X[['Size_Log']])
X['Min_Version'] = std_scaler.fit_transform(X[['Min_Version']])
X['Max_Version'] = std_scaler.fit_transform(X[['Max_Version']])
X.head()


# In[146]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)


# In[147]:


count = {}
for i in y_train:
    if i not in count:
        count[i] = 0
    count[i] = count[i] + 1
print(count)


# In[117]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_pred,y_test)*100
print("Accuracy =",round(rf_acc,2),"%")


# In[118]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit( X_train, y_train)
lr_pred = clf.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)*100
print("Accuracy =",round(lr_acc,2),"%")


# In[151]:


gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train,y_train)
gb_pred = gb_clf.predict(X_test)
gb_acc = accuracy_score(y_test,gb_pred)*100
print("Accuracy =",round(gb_acc,2),"%")


# In[148]:


from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
mlp_clf.fit( X_train, y_train)
mlp_pred = mlp_clf.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_pred)*100
print("Accuracy =",round(mlp_acc,2),"%")


# In[164]:


acc = pd.Series(index=['Random Forest','Logistic Regression','Gradient Boosting','MLP'], data=[rf_acc,lr_acc,gb_acc,mlp_acc])
plt.figure(figsize=(10,10))
s = sns.barplot(acc.index,acc.values)
plt.title('Model Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')


# In[ ]:




