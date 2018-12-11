# Churn Data Analisys

The code contained in this repository is related with the data analysis of sellers performance and the company churn data.

The initial idea was to build a model to identify trends, and then to correlate the trends with a churn event, to eventually check if the initial hypothesis, that growing trends or declining trends can lead to a churn event.

## Pre-check

First of all, we load the .csv files as pandas dataframes.

```python
seller_per = pd.read_csv('seller_performance_data.csv', sep=';', decimal=",")
churn_data = pd.read_csv('churn_data.csv', sep=';', decimal=",")
```
We can check then the head of both dataframes

for the seller_per we have

 
  | Unnamed: 0 |                         supplier_key  | report_date  |    ... |       units_ordered | units_ordered_b2b | units_refunded
  | -----------|---------------------------------------|--------------|--------|---------------------|-------------------|----------------
0 |          1 | 00179d2b-e696-4536-b530-e25ed838fae6  | 2018-10-24   |    ... |                 214 |                 0 |              0
1 |          2 | 00179d2b-e696-4536-b530-e25ed838fae6  | 2018-10-23   |    ... |                 193 |                 0 |              0
2 |          3 | 00179d2b-e696-4536-b530-e25ed838fae6  | 2018-10-22   |    ... |                 188 |                 0 |              0
3 |          4 | 00179d2b-e696-4536-b530-e25ed838fae6  | 2018-10-21   |    ... |                 160 |                 0 |              0
4 |          5 | 00179d2b-e696-4536-b530-e25ed838fae6  | 2018-10-20   |    ... |                  86 |                 0 |              0

and for churn_data we have


  | Unnamed: 0 |                         supplier_key | churn_date
  |------------|--------------------------------------|-----------  
0 |          1 | 031a13f1-5488-4ef8-a2fa-e55bb894c44e | 2018-01-18
1 |          2 | 03d96d8a-7178-4f7d-a8f2-8ef1a643ecd5 | 2016-12-09
2 |          3 | 0574ad4e-5331-4e0b-9dd5-70d8fd3ba01b | 2018-01-19
3 |          4 | 073941ba-16e0-4a1f-825e-1cd94b9d50cb | 2017-04-27
4 |          5 | 082ba7d5-7b68-46e3-8a53-fc4d52104e28 | 2018-10-02

From here we can remove the "Unnamed: 0" column, and format the dates columns to the proper type

```python
seller_per = seller_per.drop(['Unnamed: 0'], axis=1)
churn_data = churn_data.drop(['Unnamed: 0'], axis=1)

seller_per['report_date'] = pd.to_datetime(seller_per['report_date'], format='%Y-%m-%d')
churn_data['churn_date']  = pd.to_datetime(churn_data['churn_date'], format='%Y-%m-%d')

# We add columns of year and month for easier data grouping
seller_per['Year'] = seller_per['report_date'].dt.year
seller_per['Month'] = seller_per['report_date'].dt.month
```
In addition, we can check for NaN values and replace them with zeros

```python
print(seller_per.isnull().sum())
seller_per = seller_per.fillna(0)
```

## EDA

Now that we did a first check to the data, we will create a couple of dataframes, one for the clients that are still active, and one
for the clients that resignated the services:


```python
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
```

Now that the data is properly organized (we know where the data of active and non-active clients is), we can 
perform a visual inspection to identify trends or specific aspects of the data

To plot the daily sales performance of a client during its whole operative period we can write:

```python
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

```

These two figures give the time wave form of daily sales and the histogram:

![alt text](https://github.com/vjaramillo/churn_analysis/blob/master/sales_per_day_i1.png)

![alt text](https://github.com/vjaramillo/churn_analysis/blob/master/sales_per_day_hist_i1.png)

The histogram is heavily dominated by the amount of values equal to zero. Apparently, during many days he had zero sales.

This happens to many other clients, unfortunately, since this does not allows to see the proper distribution.

In order to see the distribution, we will remove those entries in which the sales were zero and plot the histogram again

```python
df3 = df2.ordered_product_sales

df3.fillna(0, inplace=True)

df3 = df3[df3 != 0]

plt.figure(2)
plt.hist(df3, bins = 100)

```

![alt text](https://github.com/vjaramillo/churn_analysis/blob/master/sales_per_day_hist_i1_wo_zeros.png)

We can see in the plot that the daily sales data is skewed. This happens to many clients, since they sell, in general, small amounts per day most of the time.
And obviously, there is no negative sales, so in the X-axis, the minimum value will be always 0 (unless we remove that data).

After exploring few of the data sets, it can be seen that the probabilistic distributions of the data, in general, are very diverse and do not fit
with a gaussian distribution, however, few of them suffer skewness.

Some data sets have not so much entries (the shortest register has entries of around 4 months), and others have registers of almost the 4 years.

With the following script, we can compare some (Max-normalized) distributions of sales between different clients (4 in this case)

```python
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

# This function actually adjusts the sub plots using the above paramters
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
```

![alt text](https://github.com/vjaramillo/churn_analysis/blob/master/comparison_distributions.png)

The first two from the top are similar (corresponding to i = 0 and i = 90), however the two from the bottom are very different (i = 230 and i = 180).

Of course, only by looking at four, nothing can be concluded, however, further analysis of the datasets shows that there is a high diversity of distributions, and most of them show some degree of skewness.

Another analysis conducted, was the trends of the data, for that analysis the following script was used:


```python
df = seller_per.drop(['supplier_key','ordered_product_sales_b2b','units_ordered_b2b','units_ordered','units_refunded','Year','Month'], axis=1)
df2 = df.loc[seller_per['supplier_key'] == supplier_keys[0]]
df2 = df2.sort_values(by='report_date')

df3 = df.loc[seller_per['supplier_key'] == supplier_keys[90]]
df3 = df3.sort_values(by='report_date') 

df4 = df.loc[seller_per['supplier_key'] == supplier_keys[230]]
df4 = df4.sort_values(by='report_date') 

df5 = df.loc[seller_per['supplier_key'] == supplier_keys[180]]
df5 = df5.sort_values(by='report_date') 

df2.index = df2.report_date 
df3.index = df3.report_date 
df4.index = df4.report_date 
df5.index = df5.report_date 

from pylab import rcParams

rcParams['figure.figsize'] = 12, 7

decomposition2 = sm.tsa.seasonal_decompose(df2.ordered_product_sales, model='additive')
decomposition3 = sm.tsa.seasonal_decompose(df3.ordered_product_sales, model='additive')
decomposition4 = sm.tsa.seasonal_decompose(df4.ordered_product_sales, model='additive')
decomposition5 = sm.tsa.seasonal_decompose(df5.ordered_product_sales, model='additive')

plt.figure(1)
trend2 = decomposition2.trend
trend3 = decomposition3.trend
trend4 = decomposition4.trend
trend5 = decomposition5.trend

plt.plot(trend2 / trend2.max())
plt.plot(trend3 / trend3.max())
plt.plot(trend4 / trend4.max())
plt.plot(trend5 / trend5.max())
plt.gca().legend(('i = 0','i = 90','i = 230','i = 180'))

```
![alt text](https://github.com/vjaramillo/churn_analysis/blob/master/comparison_trends.png)

## Analysis

With regards to the trends, the comparison is even more complicated than the comparison of probabiltiy distributions.

As can be seen in the previous figure, even Max-normalizing the data, does not allow to see specific similarities in the trend patterns. The figure only shows 4, but again, having a look to the multiple data sets, it can be seen that, even though, some datasets have similar trends, in general the trends are very diverse. In addition, not all have the same lenght, or start at the same periods of time.

In this sense, it would be very complicated to build a single model that is able to predict, with a good level of certainty, all the sales trends. This, mostly due to the differences in distribution, trends and amount of available data entries.

In the same way, the sales trends of both type of clients (active, non-active) have significant declines or inclines, and in some cases the client resigns, but in others he continues using the services, so, to consider an incline or decline (only), for predicting a churn event, could result in false positives.

Predicting the sales of the clients incline or decline should be done with a model exclusive for each client (or clusters of clients), but preferably, these models should be able to use additional categorical variables, e.g. type of product (consumer electronics, spare parts, decoration, etc.), age of seller, time active, seller capital, etc. In this way, it would be expected to have more information to feed the models, thus allowing for higher accuracy in the predictions of churn events.


## Models

Regardless of the issues identified, an initial approach was design, to identify churn event, regardles of the knowledge of sales incline/decline, but only considering monthly and daily sales.

The following code was use to extrac the data of sales per day per client:

```python
# Two arrays are created
# An array whose columns are the sales per day (1442 registers)
# An array whose columns are the sales per month (49 registers)

X_d = np.zeros((1442, len(supplier_keys)))
X_m = np.zeros((49, len(supplier_keys)))

# Columns of Year and month are added, so it can be easier to 
seller_per_tidy['Year'] = seller_per_tidy['report_date'].dt.year
seller_per_tidy['Month'] = seller_per_tidy['report_date'].dt.month

# An array to identify the active, non-active clients is also created

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

```

After getting the data, we Max-normalize it

```python
X_mnorm = normalize(X_m, axis=1, norm='max')
X_dnorm = normalize(X_d, axis=1, norm='max')
X_norm = np.hstack([X_mnorm, X_dnorm])
```
### 1layer NN

```python
X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, test_size=0.33)

model = Sequential()

n_cols = X_train.shape[1]

model.add(Dense(80, activation = 'relu', input_shape=(n_cols,), kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid', kernel_initializer='he_normal'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = 100, batch_size = 16, verbose = 0)

loss, acc = model.evaluate(X_test, y_test)
print('Test set accuracy = ', acc)

plt.figure(1)
plt.plot(history.history['acc'])
plt.figure(2)
plt.plot(history.history['loss'])
```
A manual tuning and training leads eventually to a NN with an test set accuracy of around 76% and a train set accuracy close to 100%.
This shows that the model is overfitting. The model that achieved this accuracy is saved as "model_weights766.h5".

In general, the models trained with this data (daily+monthly sales per client), the model converges very fast (under 50 epochs in some cases), showing that the data is highly correlated. Something that makes sense, given the nature of the data, previously commented.

It has to be taken into account that there is an small amount of data available. We have data from 284 clients, from which 127 have resigned the services, so the threshold to determine if the a classification system is able to detect a churn event is (1 - 127/284) * 100 =  55.3%, in this sense, the achieved model is predicting a bit more better than just guessing.

One approach to overcome this issue of overfitting, would be to have more data, or to include a bigger amount of features in for the training. Aiming for this, an statistical analysis of the monthly sales was conducted with the following code:

```python
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
```


The network was retrained using as input data:

```python
X_norm = np.hstack([X_mnorm, X_dnorm, STATS_arr])
```
however, not significant accuracy increase was perceived.

### Random Forest Classifier

As an approach to solve the issue of overfitting suffered by the NN, a random forest classifier was trained.

First, the data was prepared for the RandomForest

Each column sales_i of the dataframe, is the client sales -i months before the last report.
In total, the maximum amount of monthtly registers is 49, however this is not true for every client, so for some entries there will be zeros.

In addition, the statistical parameters of the monthly sales are included.

The parameters

```python
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
```

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(data, Target, test_size=0.3)

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 50)

# Train the model on training data
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```


### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

