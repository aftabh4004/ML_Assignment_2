import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
import random
import math


cols = ['Sex', 'Length', 'Diameter', 
        'Height', 'Whole weight', 'Shucked weight',
        'Viscera weight', 'Shell weight', 'Rings' ]

# Loading data

df = pd.read_csv('./abalone.data', delimiter=",", names=cols)
df.head()
Rings = df['Rings']
df = df.drop(['Rings'], axis = 1)




# Applying One-Hot-Encoding for the categorical variable
df = pd.get_dummies(df)
cols = df.columns



# Scaling the columns of the dataset by using StandardScaler
scaler = StandardScaler()
df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns=cols)


# ==================================== PCA =========================================== #

n_com = 0
for dim in range(2, df.shape[1]):
    n_com = dim
    pca = PCA(n_components = dim)
    pca.fit(df)
    var_p = pca.explained_variance_ratio_
    if var_p.sum() > 0.95:
        break

pca = PCA(n_components = n_com)
pca.fit(df)
var_p = pca.explained_variance_ratio_
pca_df = pca.transform(df)
print("{} componets preserve {}% of total variance".format(pca_df.shape[1], var_p.sum()))


# Creating DataFrame from the array got from PCA transform
pca_df = pd.DataFrame(pca_df)


# ================================ PCA plot ======================================== #

pca_x = pca_df[0].tolist()
pca_y = pca_df[1].tolist()
pca_z = pca_df[2].tolist()


# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(pca_x, pca_y, pca_z, color = "green")
plt.title("Scatter plot of top 3 component")
 
ax.set_xlabel('Component 0')
ax.set_ylabel('Component 1')
ax.set_zlabel('Component 2')

plt.savefig('pca_plot.png')


# ============================== K-mean Clustering Implementation ================================== #

# Helper functions

def distance(x1, x2):
    d = 0
    for i in range(len(x1)):
        d += (x1[i] - x2[i])*(x1[i] - x2[i])
    return math.sqrt(d)

def get_new_centeroids(df, clusters, k):
    new_centeroids = [[] for i in range(k)]
    for i in range(len(clusters)):
        
        for j in range(df.shape[1]):
            sum = 0
            for ci in clusters[i]:
                sum += df[j].iloc[ci]
            sum = sum / len(clusters[i])
            new_centeroids[i].append(sum)
    return new_centeroids

def get_new_clusters(df, centeroids, k):
    clusters = [[] for i in range(k)]
    
    for i in range(df.shape[0]):
        min_ = float('inf')
        si = 0
        for ki in range(k):
            d = distance(df.iloc[i].tolist(), centeroids[ki])
            if d < min_:
                min_ = d
                si = ki
        clusters[si].append(i)
    return clusters


def is_converged(centeroids, new_centeroids, tol, k):
    
    min_ = float('inf')
    for i in range(k - 1):
        for j in range(i + 1, k):
            d = distance(centeroids[i], centeroids[j])
            if d < min_:
                min_ = d
       
    
    for i in range(k):
        d = distance(centeroids[i], new_centeroids[i])
        if d > tol*min_:
            return False
    return True
        
    


# ========================== K-means Algorithm ============================== #

def kmeans_clustering(df, k, tol):
    centeroids_index = random.sample(range(0, df.shape[0]), k)
    centeroids = []
    for ci in centeroids_index :
        centeroids.append(df.iloc[ci].tolist())
    
    clusters = [[] for i in range(k)]
    
    for i in range(df.shape[0]):
        min_ = float('inf')
        si = 0
        for ki in range(k):
            d = distance(df.iloc[i].tolist(), centeroids[ki])
            if d < min_:
                min_ = d
                si = ki
        clusters[si].append(i)
    
    
    while(1):
        new_centeroids = get_new_centeroids(df, clusters, k)
        if is_converged(centeroids, new_centeroids, tol, k) == True:
            break
        
        new_clusters = get_new_clusters(df, new_centeroids, k)
        
        centeroids = new_centeroids
        clusters = new_clusters
    
    # Geting cluster labels
    
    labels = [-1] * df.shape[0]
    
    for i in range(k):
        cluster = clusters[i]
        for ci in cluster:
            labels[ci] = i
    return labels
            



# Varying k from 2 to 8 and calculating NMI for each clusters.

nmi = []
nmi_n = []
print("It might take couple of minutes...")
for i in range(2, 9):
    kmeans = cluster.KMeans(n_clusters = i, init = 'random', tol=0.05, random_state=42)
    kmeans = kmeans.fit(pca_df)
    labels = kmeans_clustering(pca_df, i, 0.05)
    nmi_i = normalized_mutual_info_score(Rings, kmeans.labels_.tolist())
    nmi_i_n = normalized_mutual_info_score(Rings,labels)
    nmi.append(nmi_i)
    nmi_n.append(nmi_i_n)


# ======================== Ploting the results ============================== #



x_n = [2, 3, 4, 5, 6, 7, 8]
y_n = nmi_n

# Creating figure
fig = plt.figure(figsize = (7, 5))
ax = plt.axes()
 
# Creating plot
ax.scatter(x_n, y_n)
plt.title("Plot for NMI with increasing number of clusters(k)")
 
ax.set_xlabel('k')
ax.set_ylabel('Normalized Mutual Info(NMI)')
plt.savefig('k-vs-nmi.png')



# Finding k for which the value of NMI is maxzmax_ = float('-inf')
max_i = 0
max_ = float('-inf')
for i in range(len(nmi_n)):
    if nmi_n[i] > max_:
        max_ = nmi_n[i]
        max_i = i


print ("for k = {} the value of NMI is  {} which is maximum".format(max_i + 2, max_))
