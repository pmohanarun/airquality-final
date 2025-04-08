#%% Importing all the packages

import time
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
import pickle
import os

#%% getting the dates

all_data = pd.read_excel(r"data/NEWTEMP_last.xlsx", parse_dates=(['Date']), index_col='Date')
cols = all_data.columns
date = all_data.index

#%% Data cleaning

# Looking for null values
print(all_data.isnull().sum())

# Dropping rows with NaN values
# all_data = all_data.dropna(axis=0)

# Null values are now removed. Print the null sum again to check.

# There are values with '*' and '#'. To remove that, I'm replacing them with NaN value and then filling those NaN values with mean of the column.
all_data = all_data.replace(dict.fromkeys(['*','**','***','#','##','###','####','#####'], 'NaN'))
print(all_data.isin(['*','**','***','#','##','###','####','#####']).sum())

# There are also values with '.', '/', '*' at the end. So stripping the data off of special charaters from the end
strip_cols = all_data.select_dtypes(object).columns
all_data[strip_cols] = all_data[strip_cols].apply(lambda x: x.astype(str).str.rstrip('/.+'))

# There are no nans now, but there are some nan values as strings. So converting all nan strings to np.nan
all_data = all_data.replace('NaN', np.nan)

# check the number of nans again
print(all_data.isnull().sum())

# Converting all the object type columns to float
# all_data.iloc[:,[4,5,6,7,8,14,15]] = all_data.iloc[:,[4,5,6,7,8,14,15]].astype(float)

# Imputing all the nan values with mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
all_data = imputer.fit_transform(all_data)
all_data = pd.DataFrame(all_data, columns=(cols), index=date)
# check the number of nans and dtypes again

print(all_data.isnull().sum())
print(all_data.info())

#%% Finding any ouliers in the dataset
#converting object type to float
all_data = all_data.astype('float')
# Using boxplots to find outliers

figure, axes = plt.subplots(2,2)
axes[0,0].boxplot(all_data['PM2.5'])
axes[0,1].boxplot(all_data['AQI'])
axes[1,0].boxplot(all_data['SO2'])
axes[1,1].boxplot(all_data['NO2'])
figure.tight_layout()

# Removing outliers

all_data['z-score'] = (all_data.AQI - all_data.AQI.mean())/all_data.AQI.std()
outliers = all_data[(all_data['z-score']<-4) | (all_data['z-score']>4)]
all_data = all_data[(all_data['z-score']>-4)& (all_data['z-score']<4)]
all_data = all_data.drop('z-score', axis=1)

#%% Resampling the data

all_data.index = pd.to_datetime(all_data.index)
all_data = all_data.sort_index()

upsampled = all_data.resample('1D').last()
plt.plot(upsampled['AQI'])

interpolated = upsampled.interpolate(method='time')
interpolated.shape
plt.plot(interpolated['AQI'])

#%% Trying k-means clustering
from sklearn.cluster import KMeans
x = all_data.loc[:,['AQI','lat','long']]

K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = all_data[['lat']]
X_axis = all_data[['long']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
# Visualize
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

#%% kmeans


location = kmeans.predict(lat_long)
location = int(location)

#%% merging the data

x = x[['AQI','cluster_label']]

all_data['cluster'] = x['cluster_label']
all_data = all_data.drop(['lat','long'], axis=1)
all_data = all_data.astype('float64')
all_data.info()

#%% saving the dataframe

# Create directory if it doesn't exist
if not os.path.exists('data/final_data_preprocessed'):
    os.makedirs('data/final_data_preprocessed')

# Save the processed data
all_data.to_csv('data/input_cluster.csv', index=True)
print("Data saved successfully to data/input_cluster.csv")


