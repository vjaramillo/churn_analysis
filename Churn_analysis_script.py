import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
import keras.initializers
from keras.optimizers import Adam

import seaborn as sns

import statsmodels.api as sm

seller_per = pd.read_csv('seller_performance_data.csv', sep=';', decimal=",")
churn_data = pd.read_csv('churn_data.csv', sep=';', decimal=",")

# First, let's have a look to the files

seller_per.head()
churn_data.head()

# Column 0 is eliminated, since does not provide valuable info

seller_per = seller_per.drop(['Unnamed: 0'], axis=1)
churn_data = churn_data.drop(['Unnamed: 0'], axis=1)

# Cast report_date column to the proper data type (datetime)

seller_per['report_date'] = pd.to_datetime(seller_per['report_date'], format='%Y-%m-%d')
churn_data['churn_date']  = pd.to_datetime(churn_data['churn_date'], format='%Y-%m-%d')

# We add columns of year and month for easier data grouping
seller_per['Year'] = seller_per['report_date'].dt.year
seller_per['Month'] = seller_per['report_date'].dt.month

# if there are null values, replace with 0
print(seller_per.isnull().sum())
seller_per = seller_per.fillna(0)

### EDA
#Now that we did a first check to the data, we will create a couple of dataframes, one for the clients that are still active, and one
#for the clients that resignated the services:

resign_seller_perf = []
active_seller_perf = seller_per.copy()

for i in range(0, len(churn_data['supplier_key'])):
  seller_churn = seller_per.loc[seller_per['supplier_key'] == churn_data['supplier_key'].iloc[i]]
  seller_churn = seller_churn.loc[seller_per['report_date'] <= churn_data['churn_date'].iloc[i]]
  active_seller_perf = active_seller_perf.drop(active_seller_perf[active_seller_perf.supplier_key == churn_data['supplier_key'].iloc[i]].index)
  resign_seller_perf.append(seller_churn)
  
resign_seller_perf = pd.concat(resign_seller_perf, axis=0)
resign_seller_perf['Active'] = 0
active_seller_perf['Active'] = 1

seller_per_tidy = pd.concat([active_seller_perf, resign_seller_perf])

supplier_keys = seller_per_tidy['supplier_key']
churned_keys = churn_data['supplier_key']
supplier_keys = supplier_keys.unique()
churned_keys = churned_keys.unique()

#Now that the data is properly organized (we know where the data of active and non-active clients is), we can 
#perform a visual inspection to identify trends or specific aspects of the data

#To plot the daily sales performance of a client during its whole operative period we can write:

df = seller_per.drop(['supplier_key','ordered_product_sales_b2b','units_ordered_b2b','units_ordered','units_refunded'], axis=1)

# There are supplier registers, so we can select any of those by changing the value of i
i = 1
df = df.loc[seller_per['supplier_key'] == supplier_keys[i]]

df.index = df.report_date 
df2 = df.sort_values(by='report_date') 

plt.figure(1)
plt.plot_date(df2.report_date, df2.ordered_product_sales, '-')
plt.figure(2)
plt.hist(df2.ordered_product_sales, bins = 100)

#The histogram is heavily dominated by the amount of values equal to zero. Apparently, during many days he had zero sales.
#This happens to many other clients, unfortunately, since this does not allows to see the proper distribution.
#In order to see the distribution, we will remove those entries in which the sales were zero and plot the histogram again

df3 = df2.ordered_product_sales

df3.fillna(0, inplace=True)

df3 = df3[df3 != 0]

plt.figure(2)
plt.hist(df3, bins = 100)

#With the following script, we can compare some max-normalized distributions of sales between different clients (4 in this case)
df = seller_per.drop(['supplier_key','ordered_product_sales_b2b','units_ordered_b2b','units_ordered','units_refunded','Year','Month'], axis=1)
df = df.sort_values(by='report_date') 

df2 = df.loc[seller_per['supplier_key'] == supplier_keys[0]]
df2_1 = df2.ordered_product_sales
df2_1.fillna(0, inplace=True)
df2_1 = df2_1[df2_1 != 0] / df2_1.max()

df3 = df.loc[seller_per['supplier_key'] == supplier_keys[90]]
df3_1 = df3.ordered_product_sales
df3_1.fillna(0, inplace=True)
df3_1 = df3_1[df3_1 != 0]/  df3_1.max()

df4 = df.loc[seller_per['supplier_key'] == supplier_keys[230]]
df4_1 = df4.ordered_product_sales
df4_1.fillna(0, inplace=True)
df4_1 = df4_1[df4_1 != 0] / df4_1.max()

df5 = df.loc[seller_per['supplier_key'] == supplier_keys[180]]
df5_1 = df5.ordered_product_sales
df5_1.fillna(0, inplace=True)
df5_1 = df5_1[df5_1 != 0] / df5_1.max()

fig, ax = plt.subplots(figsize=(12,7), ncols=2, nrows=2)

left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.9    # the top of the subplots of the figure
wspace =  .5     # the amount of width reserved for blank space between subplots
hspace =  1.1    # the amount of height reserved for white space between subplots

plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  top, 
    wspace  =  wspace, 
    hspace  =  hspace
)

sns.distplot(df2_1,  ax=ax[0][0])
sns.distplot(df3_1,  ax=ax[0][1])
sns.distplot(df4_1,  ax=ax[1][0])
sns.distplot(df5_1,  ax=ax[1][1])

#sales_per_month['Active']

# Two new DataFrames are created, one with the data of sellers that are still active, and the second with the data of the sellers that resignated (churn)
resign_seller_perf = []
active_seller_perf = seller_per.copy()

# For the dataframe of clients that resignated, only the data before the resignation date is taken into account.

for i in range(0, len(churn_data['supplier_key'])):
  seller_churn = seller_per.loc[seller_per['supplier_key'] == churn_data['supplier_key'].iloc[i]]
  seller_churn = seller_churn.loc[seller_per['report_date'] <= churn_data['churn_date'].iloc[i]]
  active_seller_perf = active_seller_perf.drop(active_seller_perf[active_seller_perf.supplier_key == churn_data['supplier_key'].iloc[i]].index)
  resign_seller_perf.append(seller_churn)
  
resign_seller_perf = pd.concat(resign_seller_perf, axis=0)
resign_seller_perf['Active'] = 0
active_seller_perf['Active'] = 1

seller_per_tidy = pd.concat([active_seller_perf, resign_seller_perf])

supplier_keys = seller_per_tidy['supplier_key']
churned_keys = churn_data['supplier_key']

# A couple of vectors of suppliers_keys are created, both contain unique values only
supplier_keys = supplier_keys.unique()
churned_keys = churned_keys.unique()

# Two arrays are created
# An array whose columns are the sales per day (1442 registers)
# An array whose columns are the sales per month (49 registers)
# An array of descriptive statistics data

X_d = np.zeros((1442, len(supplier_keys)))
X_m = np.zeros((49, len(supplier_keys)))

Y   = np.ones((len(supplier_keys),1))

for i in range(0, len(supplier_keys)):
  s_d = seller_per_tidy.loc[seller_per_tidy['supplier_key'] == supplier_keys[i]].ordered_product_sales.tolist()
  s_m = seller_per_tidy.loc[seller_per_tidy['supplier_key'] == supplier_keys[i]].groupby(['Year','Month']).sum().ordered_product_sales.tolist()
      
  s_d = np.array(s_d)
  s_m = np.array(s_m)
  
  s_d.shape = (len(s_d),1)
  s_m.shape = (len(s_m), 1)  
  
  s_d = np.flip(s_d,0)
  s_m = np.flip(s_m,0)
  
  s_d_norm = normalize(s_d, axis=1, norm='max')
  s_m_norm = normalize(s_m, axis=1, norm='max')

  if (supplier_keys[i] in churned_keys):
    Y[i] = 0
    
  X_d[ X_d.shape[0] - len(s_d) : X_d.shape[0] , [i]] = s_d
  X_m[ X_m.shape[0] - len(s_m) : X_m.shape[0] , [i]] = s_m
  
X_d = X_d.transpose()
X_m = X_m.transpose()


"""**Data preparation for 1Layer NN**"""

X_mnorm = normalize(X_m, axis=1, norm='max')
X_dnorm = normalize(X_d, axis=1, norm='max')
X_norm = np.hstack([X_mnorm, X_dnorm])

"""**1 Layer NN**"""

X_train, X_test, y_train, y_test = train_test_split(X_mnorm, Y, test_size=0.33)

model = Sequential()

n_cols = X_train.shape[1]

model.add(Dense(10, activation = 'relu', input_shape=(n_cols,), kernel_initializer='he_normal'))
#model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation = 'sigmoid', kernel_initializer='he_normal'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = 500, batch_size = 16, verbose = 0)

loss, acc = model.evaluate(X_test, y_test)
print('Test set accuracy = ', acc)

#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Test Accuracy: {} \n Test Error: {}".format(scores[1], 100-scores[1]*100))
plt.figure(1)
plt.plot(history.history['acc'])
plt.figure(2)
plt.plot(history.history['loss'])

# statistical variables extraction

stats_var_per_day = seller_per_tidy.drop(['supplier_key','ordered_product_sales_b2b','units_ordered_b2b','units_ordered','units_refunded','Year','Month','Active'], axis=1)
STATS_arr = []

for i in range(0, len(supplier_keys)):
  df = stats_var_per_day.loc[seller_per_tidy['supplier_key'] == supplier_keys[i]]
  df = df.ordered_product_sales / df.ordered_product_sales.max()
  df = df[df != 0]
  
  skew = df.skew()
  stats = df.describe()
    
  stats_v = np.array([stats[1],stats[2],stats[4],stats[5],stats[6],skew])
  stats_v.shape = (1,6)
  
  if i == 0:
    STATS_arr = stats_v

  else:
    STATS_arr = np.vstack([STATS_arr, stats_v])
    
# Statistical data extraction
# Each value sales_i, is the client sales -i months before the last report

Target = pd.DataFrame(Y,columns=['Active_client'])

data = pd.DataFrame({
    'sales_1':X_m[:,-1],
    'sales_2':X_m[:,-2],
    'sales_3':X_m[:,-3],
    'sales_4':X_m[:,-4],
    'sales_5':X_m[:,-5],
    'sales_6':X_m[:,-6],
    'sales_7':X_m[:,-7],
    'sales_8':X_m[:,-8],
    'sales_9':X_m[:,-9],
    'sales_10':X_m[:,-10],
    'sales_11':X_m[:,-11],
    'sales_12':X_m[:,-12],
    'sales_13':X_m[:,-13],
    'sales_14':X_m[:,-14],
    'sales_15':X_m[:,-15],
    'sales_16':X_m[:,-16],
    'sales_17':X_m[:,-17],
    'sales_18':X_m[:,-18],
    'sales_19':X_m[:,-19],
    'sales_20':X_m[:,-20],
    'sales_21':X_m[:,-21],
    'sales_22':X_m[:,-22],
    'sales_23':X_m[:,-23],
    'Mean': STATS_arr[:,0], 
    'Std':  STATS_arr[:,1],
    '25%':STATS_arr[:,2],
    '50%':STATS_arr[:,3],
    '75%':STATS_arr[:,4],
    'skew': STATS_arr[:,5]
})

"""**Random Forest**"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(data, Target, test_size=0.3)

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 50)

# Train the model on training data
rf.fit(X_train, y_train.values.ravel())
y_pred = rf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))