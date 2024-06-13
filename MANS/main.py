import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load your data
df = pd.read_csv('C:/Users/Luke/Desktop/projeto/MANS/dataset/marketing_campaign.csv')

# Convert categorical columns to numeric using LabelEncoder
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Convert the DataFrame to a NumPy array
df = df.values

# wcss = []  # Within-Cluster-Sum-of-Squares (WCSS)
# for i in range(1, 11):  # change the range according to your data
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(df)
#     wcss.append(kmeans.inertia_)  # inertia_ is the WCSS for that model

# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

silhouette_scores = []  # Silhouette scores for different numbers of clusters

for i in range(2, 11):  # Silhouette score is not defined for a single cluster
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df)
    score = silhouette_score(df, kmeans.labels_)
    silhouette_scores.append(score)

plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

df['cluster'] = kmeans.labels_

feature_stats = df.groupby('cluster')['MntWines'].describe()
print(feature_stats)

# Create a K-means model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Fit the model to your data
kmeans.fit(df)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the cluster centers
centers = kmeans.cluster_centers_

# Print the results
print("Cluster labels:", labels)
print("Cluster centers:", centers)