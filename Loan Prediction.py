#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the Libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="white", color_codes=True)


# In[6]:


df = pd.read_csv("C:\\Users\\lenovo\\loan_data_set.csv")
df.head(10)                                  


# In[7]:


# Describing entire Dataset 

df.describe()


# In[8]:


# Obtaining all the information 

df.info()


# In[9]:


# Checking rows and Columns of dataset

df.shape


# In[10]:


# checking the existance of Null values in each column

df.isnull().any()


# In[11]:


# Checking the number of Null Values in each column

df.isnull().sum()


# In[12]:


# A way to get info of a particular column

df[['Gender']].info()


# In[13]:


df.head(10)


# In[14]:


# To check the number of outcomes in a particular column (Represented in form of array)

df["Property_Area"].unique()


# In[15]:


# Count of occurance of each outcome of a column

df["Property_Area"].value_counts()


# In[16]:


# This method is the fastest way to clean the data but it has a disadvantage of loosing important data

df_loan = df.dropna()
df_loan.info()


# In[17]:


# Dependents column has 599 Non-Null values

df.info()


# In[25]:


## Data Preprocessing


# In[26]:


# All Non-null values are replaced by digit 1 in Dependents column.. So it has 0 Null-values now

df['Dependents'].fillna(1,inplace=True)
df.info()


# In[27]:


# All null values in LoanAmount section is filled by the mean value of its own column to obtain 614 Non-null values

df['LoanAmount'].fillna(df.LoanAmount.mean(),inplace = True)
df.info()


# In[28]:


df.head(10)


# In[29]:


# Creating a duplicate column of a column with binary outcomes in character form and appending it in the table as shown.  

Value_Mapping = {'Yes' : 1, 'No' : 0}
df['Married_Section'] = df['Married'].map(Value_Mapping)
df.head(5)


# In[30]:


# Creating a duplicate column of a column with binary outcomes in character form and appending it in the table as shown.

Value_Mapping1 = {'Male' : 1, 'Female' : 0}
df['Gender_Section'] = df['Gender'].map(Value_Mapping1)
df.head(5)


# In[31]:


df["Education"].unique()


# In[32]:


# Creating a duplicate column of a column with binary outcomes in character form and appending it in the table as shown.

Value_Mapping2 = {'Graduate' : 1, 'Not Graduate' : 0}
df['Edu_Section'] = df['Education'].map(Value_Mapping2)
df.head(5)


# In[33]:


df.info()


# In[34]:


# Filling all the empty spaces of the following columns having binary outcomes in integer form.

df["Married_Section"].fillna(df.Married_Section.mean(),inplace=True) 

df["Gender_Section"].fillna(df.Gender_Section.mean(),inplace=True)

df["Loan_Amount_Term"].fillna(df.Loan_Amount_Term.mean(),inplace=True)

df["Credit_History"].fillna(df.Credit_History.mean(),inplace=True)
df.info()


# In[35]:


# Creating a duplicate column of a column with binary outcomes in character form and appending it in the table as shown.

Value_Mapping3 = {'Yes' : 1, 'No' : 0}
df['Employed_Section'] = df['Self_Employed'].map(Value_Mapping3)
df.head(5)


# In[36]:


df.info()


# In[37]:


df["Employed_Section"].fillna(df.Employed_Section.mean(),inplace=True)
df.info()


# In[38]:


# Filling the empty spaces of the column having more than 2 outcomes in character form.

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["Property_Section"] = lb_make.fit_transform(df["Property_Area"])
df.head(5)


# In[39]:


Value_Mapping4 = {'Y' : 1, 'N' : 0}
df['Loan_Section'] = df['Loan_Status'].map(Value_Mapping4)
df.head(5)

# Dependents Column is not used in data representation and algorithm implementation


# In[40]:


##Data Representation


# In[41]:


# Using SCATTER REPRESENTATION
# Imported libraries... x-axis: loan_status, y_axis: Loan_Amount and representing in terms of Gender_Section

sns.FacetGrid(df,hue="Gender_Section",size=4) .map(plt.scatter,"Loan_Status","LoanAmount") .add_legend()
plt.show()


# In[42]:


# x-axis: ApplicationIncome, y_axis: CoapplicantIncome and representing in terms of Property_Section

sns.FacetGrid(df,hue="Property_Section",size=4) .map(plt.scatter,"ApplicantIncome","CoapplicantIncome") .add_legend()
plt.show()


# In[43]:


# Using HISTOGRAM REPRESENTATION
# x-axis: loan figures, y_axis: count, Title: Loan taken by Customers

plt.figure(figsize = (10,7)) 
x = df["LoanAmount"] 
plt.hist(x, bins = 30, color = "pink") 
plt.title("Loan taken by Customers") 
plt.xlabel("Loan Figures") 
plt.ylabel("Count")


# In[44]:


#USING BOXPLOT REPRESENTATION
# x-axis: Property_Area with 3 types of outcomes, y-axis: Gender_Section with 3 types of outcomes

sns.boxplot(x="Property_Area", y="Gender_Section", data=df)


# In[49]:


# x-axis: Married_Section with 3 types of outcomes, y-axis: ApplicantIncome with all outcomes

sns.boxplot(x="Married_Section", y="ApplicantIncome", data=df)


# In[108]:


df_temp=df[df["Education"]== "Graduate"]
df_temp["Self_Employed"].hist()


# In[103]:


sns.FacetGrid(df, hue="Credit_History", size=6).map(sns.kdeplot, "CoapplicantIncome").add_legend()


# In[52]:


##Correlation Matrix


# In[54]:


# Correlation Matrix of the columns given below

cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Married_Section',
        'Gender_Section','Edu_Section','Employed_Section','Property_Section']
f, ax = plt.subplots(figsize=(10, 7))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.show()


# In[56]:


##Algorithm Implementation
##Logistic Reggression


# In[57]:


# Importing Libraries and classes

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[58]:


# X is the input and Y is the output

X=df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Married_Section',
        'Gender_Section','Edu_Section','Employed_Section','Property_Section']].values
y=df[["Loan_Section"]].values


# In[59]:


# Importing Libraries and classes
# Dividing the data in 7:3 Ratio for Training and Testing respectively

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[60]:


# Training the Model

model.fit(X_train,y_train)


# In[61]:


# Training Accuracy

model.score(X_train,y_train)


# In[62]:


#Testing Accuracy

model.score(X_test,y_test)


# In[63]:


# Y contains all the outputs and X contains all the inputs. We will test on the machine if it gives to expected Output for the 
# corresponding Inputs.

expected = y_test
predicted = model.predict(X_test)


# In[64]:


# Importing Libraries and class 

from sklearn import metrics


# In[65]:


#  Printing the Report

print(metrics.classification_report(expected, predicted))


# In[66]:


# Out of 51 'Y' outcomes, 23 were right and 28 were wrong similarly, for'N', 131 were right and 3 were wrong.

print(metrics.confusion_matrix(expected, predicted))


# In[67]:


####Support Vector Machine


# In[68]:


# Importing class and libraries

from sklearn.svm import SVC
model = SVC()


# In[69]:


# Training the Model

model.fit(X_train,y_train)


# In[80]:


# Accuracy of the model in training

model.score(X_train,y_train)


# In[81]:


# Accuracy of the model in Testing

model.score(X_test,y_test)


# In[82]:


# Importing Libraries and Classes

from sklearn import metrics


# In[73]:


# Obtaining Report

print(metrics.classification_report(expected, predicted))


# In[74]:


# Output in the form of count

print(metrics.confusion_matrix(expected, predicted))


# In[75]:


##Random Forest


# In[76]:


# Importing libraries and classes

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[77]:


# Training the model

model.fit(X_train,y_train)


# In[78]:


# Accuracy in training the model

model.score(X_train,y_train)


# In[79]:


#Accuracy in testing the model

model.score(X_test,y_test)


# In[89]:


# Y contains all the outputs and X contains all the inputs. We will test on the machine if it gives to expected Output for the 
# corresponding Inputs.

expected = y_test
predicted = model.predict(X_test)


# In[90]:


# Generating Report

print(metrics.classification_report(expected, predicted))


# In[93]:


# Output in the form of Matrix

print(metrics.confusion_matrix(expected, predicted))


# In[86]:


##Decision Tree


# In[94]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[109]:


model.fit(X_train,y_train)


# In[110]:


model.score(X_train,y_train)


# In[111]:


model.score(X_test,y_test)


# In[112]:


expected = y_test
predicted = model.predict(X_test)


# In[113]:


print(metrics.classification_report(expected, predicted))


# In[114]:


print(metrics.confusion_matrix(expected, predicted))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




