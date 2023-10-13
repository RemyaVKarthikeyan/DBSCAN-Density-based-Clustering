#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#run before importing Kmeans
import os
os.environ["OMP_NUM_THREADS"]='1'

#importing the dataset
dataset=pd.read_csv("Mall_Customers.csv")
dataset

#importing StandardScaler from scikit-learn library
from sklearn.preprocessing import StandardScaler

# select all rows (:) and only the columns at index 3 and 4
X=dataset.iloc[:,[3,4]].values

# creating an instance of the StandardScaler class and storing it in the variable sc_X
sc_X=StandardScaler()

#standardizing the values stored in X (mean =0, sd =1)
X=sc_X.fit_transform(X)
X

from sklearn.neighbors import NearestNeighbors

#find the two nearest neighbors for each data point 
# the first nearest neighbor is the data point itself
neighbours=NearestNeighbors(n_neighbors=2)

#distances and indices of the nearest neighbors for each data point in the dataset X.
distances,indices=neighbours.fit(X).kneighbors(X)

#the distances to the second nearest neighbors is determined and assigns them to the distances variable. 
#The second nearest neighbor is chosen because the nearest neighbor is the data point itself (distance of 0).
distances=distances[:,1]

#sorts the distances, which is a 1D array, from the smallest to the largest values.
distances=np.sort(distances,axis=0)

#plot of the sorted distances
plt.title('Finding the epsilon distance')
plt.xlabel('data points')
plt.ylabel('epsilon')
plt.plot(distances)

from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=0.25, min_samples=5)
y_dbscan=dbscan.fit_predict(X)

#inspect the array to identify the number of clusters
y_dbscan

#-1 indicate those points which haven’t been assigned a cluster as they are considered noise

#Visualizing the clusters

plt.figure(figsize=(15,6))
plt.scatter(X[y_dbscan==0,0],X[y_dbscan==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_dbscan==1,0],X[y_dbscan==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_dbscan==2,0],X[y_dbscan==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_dbscan==3,0],X[y_dbscan==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(X[y_dbscan==4,0],X[y_dbscan==4,1],s=100,c='magenta',label='Cluster 5')
plt.scatter(X[y_dbscan==5,0],X[y_dbscan==5,1],s=100,c='brown',label='Cluster 6')
plt.scatter(X[y_dbscan==-1,0],X[y_dbscan==-1,1],s=100,c='black',label='Noise')
plt.title('Cluster of customers')
plt.xlabel('Annual Income(Scaled)')
plt.ylabel('Spending Income(Scaled)')
plt.legend()
plt.show()


# In[ ]:




