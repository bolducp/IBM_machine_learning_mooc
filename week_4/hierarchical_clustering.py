import os
try:
	os.chdir(os.path.join(os.getcwd(), 'week_4'))
	print(os.getcwd())
except:
	pass
from IPython import get_ipython
#%%
import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')

X1, y1 = make_blobs(n_samples=100, centers=6, cluster_std=0.9)

plt.scatter(X1[:, 0], X1[:, 1], marker='o') 
#%%
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'complete')
#%%
agglom.fit(X1,y1)

#%%
# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6,4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)

# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
    
# Remove the x ticks, y ticks, x and y axis
# plt.xticks([])
# plt.yticks([])
#plt.axis('off')



# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()

#%%
dist_matrix = distance_matrix(X1,X1) 
print(dist_matrix)

#%%
Z = hierarchy.linkage(dist_matrix, 'complete')

#%%
dendro = hierarchy.dendrogram(Z)

#%%
Z = hierarchy.linkage(dist_matrix, 'average')
dendro = hierarchy.dendrogram(Z)

#%%
get_ipython().system('wget -O cars_clus.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv')

filename = 'cars_clus.csv'

#Read csv
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape)
pdf.head(5)

# The feature sets include  price in thousands (price), engine size (engine_s), horsepower (horsepow), wheelbase (wheelbas), width (width), length (length), curb weight (curb_wgt), fuel capacity (fuel_cap) and fuel efficiency (mpg).
# lets simply clear the dataset by dropping the rows that have null value:

#%%
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)

#%% [markdown]
# ### Feature selection
# Lets select our feature set:
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

#%% [markdown]
# ### Normalization
# Now we can normalize the feature set. __MinMaxScaler__ transforms features by scaling each feature to a given range. It is by default (0, 1). That is, this estimator scales and translates each feature individually such that it is between zero and one.
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx[0:5]

#%% [markdown]
# <h2 id="clustering_using_scipy">Clustering using Scipy</h2>
# In this part we use Scipy package to cluster the dataset:  
# First, we calculate the distance matrix. 

#%%
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

#%% [markdown]
# In agglomerative clustering, at each iteration, the algorithm must update the distance matrix to reflect the distance of the newly formed cluster with the remaining clusters in the forest. 
# The following methods are supported in Scipy for calculating the distance between the newly formed cluster and each:
#     - single
#     - complete
#     - average
#     - weighted
#     - centroid
#     
#     
# We use __complete__ for our case, but feel free to change it to see how the results change.

#%%
import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')

#%% [markdown]
# Essentially, Hierarchical clustering does not require a pre-specified number of clusters. However, in some applications we want a partition of disjoint clusters just as in flat clustering.
# So you can use a cutting line:

#%%
from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
clusters

#%% [markdown]
# Also, you can determine the number of clusters directly:

#%%
from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
clusters

# Now, plot the dendrogram:
#%%
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

#%%
dist_matrix = distance_matrix(feature_mtx,feature_mtx) 
print(dist_matrix)

#%% [markdown]
# Now, we can use the 'AgglomerativeClustering' function from scikit-learn library to cluster the dataset. The AgglomerativeClustering performs a hierarchical clustering using a bottom up approach. The linkage criteria determines the metric used for the merge strategy:
# 
# - Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
# - Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
# - Average linkage minimizes the average of the distances between all observations of pairs of clusters.

#%%
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
agglom.labels_

#%% [markdown]
# And, we can add a new field to our dataframe to show the cluster of each row:

#%%
pdf['cluster_'] = agglom.labels_
pdf.head(50)


#%%
import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.resale[i], subset.price[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.resale, subset.price, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
    plt.scatter(subset.resale, subset.price)
plt.legend()
plt.title('Clusters')
plt.xlabel('resale')
plt.ylabel('price')

#%% [markdown]
# As you can see, we are seeing the distribution of each cluster using the scatter plot, but it is not very clear where is the centroid of each cluster. Moreover, there are 2 types of vehicles in our dataset, "truck" (value of 1 in the type column) and "car" (value of 1 in the type column). So, we use them to distinguish the classes, and summarize the cluster. First we count the number of cases in each group:

#%%
pdf.groupby(['cluster_','type'])['cluster_'].count()

# Now we can look at the characteristics of each cluster:

#%%
agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
agg_cars

#%% [markdown]
# 
# It is obvious that we have 3 main clusters with the majority of vehicles in those.
# 
# __Cars__:
# - Cluster 1: with almost high mpg, and low in horsepower.
# - Cluster 2: with good mpg and horsepower, but higher price than average.
# - Cluster 3: with low mpg, high horsepower, highest price.
#     
#     
#     
# __Trucks__:
# - Cluster 1: with almost highest mpg among trucks, and lowest in horsepower and price.
# - Cluster 2: with almost low mpg and medium horsepower, but higher price than average.
# - Cluster 3: with good mpg and horsepower, low price.
# 
# 
# Please notice that we did not use __type__ , and __price__ of cars in the clustering process, but Hierarchical clustering could forge the clusters and discriminate them with quite high accuracy.

#%%
plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')