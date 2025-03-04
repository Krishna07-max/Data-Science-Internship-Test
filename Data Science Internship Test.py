#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing basic necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ###### Loading the datasets

# In[2]:


#Load the dataset using Pandas.
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_stores = pd.read_csv('stores.csv')
df_oil = pd.read_csv('oil.csv')
df_holidays = pd.read_csv('holidays_events.csv')


# ###### Analysing and cleaning the loaded datasets

# In[3]:


# Get first 5 rows of train dataset
df_train.head()


# In[4]:


# Summary of train dataset
df_train.info()


# In[5]:


# Check the null values in train dataset
df_train.isnull().sum()


# In[6]:


# Check the duplicate values in train dataset
df_train.duplicated().sum()


# In[7]:


# Get first 5 rows of test dataset
df_test.head()


# In[8]:


# Summary of test dataset
df_test.info()


# In[10]:


# Check for duplicates in test dataset
df_test.duplicated().sum()


# In[11]:


# Get first 5 rows of test dataset
df_oil.head()


# In[12]:


# Summary of oil dataset
df_oil.info()


# In[13]:


# Check the null values in train dataset
df_oil.isnull().sum()


# In[16]:


#- Handle missing values in oil prices by filling gaps with interpolation.
df_oil['dcoilwtico'] = df_oil['dcoilwtico'].interpolate(method = 'linear')


# In[17]:


# Recheck the missing values after interpolation
df_oil.isnull().sum()


# In[18]:


# Drop the remaining null values after interpolation
df_oil = df_oil.dropna()


# In[20]:


# Check for duplicate values in oil dataset
df_oil.duplicated().sum()


# In[21]:


# Get first 5 rows of stores dataset
df_stores.head()


# In[22]:


# Get a quick summary of store dataset
df_stores.info()


# In[24]:


# Check for duplicates in store dataset
df_stores.duplicated().sum()


# In[25]:


# Get first 5 rows of holidays dataset
df_holidays.head()


# In[26]:


# Get a quick summary of store dataset
df_holidays.info()


# In[27]:


# Check for duplicates in holidays dataset
df_holidays.duplicated().sum()


# ###### I found the missing values in oil dataset, so as per the instruction I have handled those values with 'interpolation' method.There were no null values or duplicate records observed in other datasets. 

# In[28]:


# Convert date columns to proper datetime formats.
df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])
df_oil['date'] = pd.to_datetime(df_oil['date'])
df_holidays['date'] = pd.to_datetime(df_holidays['date'])


# ###### Converted date columns of the datasets from 'object' to 'datetime' 

# In[29]:


#Merge data from stores.csv, oil.csv, and holidays_events.csv into the main dataset.
   # Here 'train' & 'test' both are main datasets


# In[30]:


df_train = df_train.merge(df_oil, on = 'date', how = 'left')
df_test = df_test.merge(df_oil, on = 'date', how = 'left')


# In[31]:


df_train = df_train.merge(df_holidays, on = 'date', how = 'left')
df_test = df_test.merge(df_holidays, on = 'date', how = 'left')


# In[32]:


df_train = df_train.merge(df_stores, on = 'store_nbr', how = 'left')
df_test= df_test.merge(df_stores, on = 'store_nbr', how = 'left')


# In[34]:


# Examined the null values after the merge in train dataset
df_train.isnull().sum()


# In[35]:


# Drop the null values in train dataset
df_train = df_train.dropna()


# In[36]:


# Examined the null values after the merge in test dataset
df_test.isnull().sum()


# In[37]:


# Drop the null values in test dataset
df_test = df_test.dropna()


# ###### 2. Feature Engineering- Time-based Features:
#   - Extract day, week, month, year, and day of the week.
#   - Identify seasonal trends (e.g., are sales higher in December?).

# In[39]:


#Extract day, week, month, year, and day of the week in train dataset
df_train['day'] = df_train['date'].dt.day
df_train['week'] = df_train['date'].dt.isocalendar().week
df_train['month'] = df_train['date'].dt.month
df_train['year'] = df_train['date'].dt.year
df_train['weekday'] = df_train['date'].dt.weekday


# In[40]:


#Extract day, week, month, year, and day of the week in test dataset
df_test['day'] = df_test['date'].dt.day
df_test['week'] = df_test['date'].dt.isocalendar().week
df_test['month'] = df_test['date'].dt.month
df_test['year'] = df_test['date'].dt.year
df_test['weekday'] = df_test['date'].dt.weekday


# In[41]:


df_train.head()


# In[42]:


#Identify seasonal trends (e.g., are sales higher in December?).

monthly_sales_trend = df_train.groupby('month')['sales'].mean()


# In[43]:


# Visualization of the monthly trend
plt.figure(figsize = (20,10))
plt.plot(monthly_sales_trend)
plt.xlabel('Months')
plt.ylabel('Average Sales')
plt.title('Monthly sales trend')
plt.grid()
plt.show()


# ###### The graph shows that the month 12(december) shows highest sales and 6(june) is the lowest.

#  Event-based Features:
#   - Create binary flags for holidays, promotions, and economic events.
#   - Identify if a day is a government payday (15th and last day of the month).
#   - Consider earthquake impact (April 16, 2016) as a separate feature.

# In[46]:


# Create binary flags for holidays, promotions, and economic events.


# In[45]:


df_train['isweekoff'] = df_train['weekday'].isin([5,6]).astype(int)


# In[47]:


df_train.head()


# ###### I think the last 2 queries are irrelavent and I found that the queries and data are unmatched. 
# The queries are:
# (Identify if a day is a government payday (15th and last day of the month).
# Consider earthquake impact (April 16, 2016) as a separate feature.)

#  Rolling Statistics:
#   - Compute moving averages and rolling standard deviations for past sales.
#   - Include lagged features (e.g., sales from the previous week, previous month).

# In[48]:


#Compute moving averages and rolling standard deviations for past sales.
df_train['rollmean'] = df_train.groupby(['store_nbr', 'family'])['sales'].transform(lambda X : X.rolling(window = 7, min_periods = 1).mean())


# In[49]:


df_train['rollstd'] = df_train.groupby(['store_nbr', 'family'])['sales'].transform(lambda X : X.rolling(window = 7, min_periods = 1).std())


# In[50]:


# Lagged sales for week and month
df_train['laged_sale_7'] = df_train.groupby(['store_nbr', 'family'])['sales'].shift(7)


# In[51]:


df_train['laged_sale_30'] = df_train.groupby(['store_nbr', 'family'])['sales'].shift(30)


#  Store-Specific Aggregations:
#   - Compute average sales per store type.
#   - Identify top-selling product families per cluster.

# In[52]:


#Compute average sales per store type.
average_sales_type = df_train.groupby('type_x')['sales'].mean().reset_index()


# In[53]:


average_sales_type


# In[54]:


average_sales_type = average_sales_type.rename(columns = {'sales':'avg_sales_per_store'})


# In[55]:


df_train = df_train.merge(average_sales_type, on = 'type_x', how = 'left')


# Exploratory Data Analysis (EDA)
# - Visualize sales trends over time.
# - Analyze sales before and after holidays and promotions.
# - Check correlations between oil prices and sales trends.
# - Identify anomalies in the data.

# In[57]:


# Visualization of sales trend over time

Daily_sales = df_train.groupby('date')['sales'].sum()

plt.figure(figsize = (20,10))
plt.plot(Daily_sales)
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('sales trend over time')
plt.grid()
plt.show()


# ###### It shows that at the year end the sales are very high around the month of November and December

# In[59]:


#Check correlations between oil prices and sales trends.

plt.figure(figsize = (20,10))
sns.scatterplot(x =df_train['dcoilwtico'] , y = df_train['sales'] )
plt.xlabel('Oil Price')
plt.ylabel('Sales')
plt.title('correlations between oil prices and sales trends')
plt.grid()
plt.show()


# In[63]:


corr = df_train[['dcoilwtico', 'sales']].corr()
print(corr)


# ###### The above graph shows litrally "no correlation" or slightly negative correlation between 'oil price' and 'sales' column

# In[65]:


#Identify anomalies in the data.
df_train['sales_diff'] =  df_train['sales'].diff()

plt.figure(figsize = (20,10))
plt.plot(df_train['date'], df_train['sales_diff'], color = 'blue')
plt.axhline(y =0, color ='black', linestyle = '--')
plt.xlabel('date')
plt.ylabel('Sales Difference')
plt.title('Daily sales Flactuations(anomalies)')
plt.grid()
plt.show()


# ###### There are high flactuations between 01-2016 and 07-2016

# In[66]:


df_train.info()


# ###### I found some null values in 'laged_sale_30 ', 'laged_sale_7', 'rollstd' columns after Feature Engineering & EDA, so the dataset was large and there were very small amount of null values so I decided to drop them.

# In[67]:


df_train = df_train.dropna()


# ###### Part 2: Model Selection, Forecasting, and Evaluation (Day 2)
#  1. Model Training
#  Train at least five different time series forecasting models:
#  - Baseline Model (Naïve Forecasting) 
#  - Assume future sales = previous sales.
#  - ARIMA (AutoRegressive Integrated Moving Average) - A traditional time series model.
#  - Random Forest Regressor - Tree-based model to capture non-linear relationships.
#  - XGBoost or LightGBM - Gradient boosting models to improve accuracy.
#  - LSTM (Long Short-Term Memory Neural Network) - A deep learning-based forecasting model.
#  Bonus Challenge: If comfortable, implement a Prophet model for handling seasonality.

# Random Forest Regressor - Tree-based model to capture non-linear relationships.

# In[71]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[73]:


# Splitting input and output coulms
X = df_train.drop(columns = 'sales')
y = df_train['sales']


# In[75]:


# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[77]:


rf = RandomForestRegressor(n_estimators = 100, random_state = 42)


# In[95]:


rf.fit(X_train, y_train)


# In[ ]:





# In[81]:


pip install xgboost


# In[83]:


from xgboost import XBGRegressor


# In[84]:


from statsmodels.tsa.arima.model import ARIMA


# In[90]:


model = ARIMA(df_train['sales'], order = (1,1,1))
model_fit = model.fit()


# In[92]:


forecast = model_fit.forecast(steps =15)
print(forecast)


# In[94]:


from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:




