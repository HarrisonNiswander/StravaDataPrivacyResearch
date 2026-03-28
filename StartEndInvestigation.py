#Harrison Niswander
#Honor's Project | CS 593 Ethics of AI
#Faculty Advisor: Professor Rusert
#3-28-26

import pandas as pd
import ast

#Define Athlete (by letter)
athlete = "A"

#import athlete data
df = pd.read_csv('Athlete_Data/Athlete' + athlete + '.csv')

# Convert starting longitude and latitude into seperate columns
# Convert string → list
df['start_latlng'] = df['start_latlng'].apply(ast.literal_eval)

# Split into separate columns
df['start_lat'] = df['start_latlng'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
df['start_lng'] = df['start_latlng'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)

# Drop missing values
startCord = df[['start_lat', 'start_lng']].dropna()

#import DBSCAN (clustering Algorithm)
from sklearn.cluster import DBSCAN

import numpy as np

startCord_rad = np.radians(startCord)

# DBSCAN with haversine metric -> convert to radians so we can do distance
kms_per_radian = 6371.0088
epsilon = 0.5 / kms_per_radian      #Radius for Clustering

db = DBSCAN(
    eps=epsilon,
    min_samples=20,                 #min number of activities to be considered cluster
    algorithm='ball_tree',
    metric='haversine'
).fit(startCord_rad)

startCord['cluster'] = db.labels_

#determine meaningful clusters
startCord['cluster'].value_counts()

#find the center of the clusters
cluster_centers = startCord.groupby('cluster')[['start_lat','start_lng']].mean()
print(cluster_centers)

#print map of clusters with noise seperated
# import matplotlib.pyplot as plt

# noise = startCord[startCord['cluster'] == -1]
# clusters = startCord[startCord['cluster'] != -1]

# plt.figure(figsize=(8,6))

# plt.scatter(clusters['start_lng'], clusters['start_lat'], c=clusters['cluster'])
# plt.scatter(noise['start_lng'], noise['start_lat'], marker='x')  # noise as X

# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Clusters vs Noise")

# plt.show()


#use Folium to create real map of clusters
import folium

#center map around data
center_lat = startCord['start_lat'].mean()
center_lng = startCord['start_lng'].mean()

athleteMap = folium.Map(location=[center_lat, center_lng], zoom_start=12)

#define colors for clusters
import random

uniqueClusters = startCord['cluster'].unique()

clusterColors = {
    cluster: "#{:06x}".format(random.randint(0, 0xFFFFFF))
    for cluster in uniqueClusters if cluster != -1
}

#plot points
for _, row in startCord.iterrows():
    cluster = row['cluster']
    
    if cluster == -1:
        color = 'gray'  # noise
    else:
        color = clusterColors[cluster]
    
    folium.CircleMarker(
        location=[row['start_lat'], row['start_lng']],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.7
    ).add_to(athleteMap)

#highlight where the center of clusters are at
centers = startCord[startCord['cluster'] != -1].groupby('cluster')[['start_lat','start_lng']].mean()

for cluster, row in centers.iterrows():
    folium.Marker(
        location=[row['start_lat'], row['start_lng']],
        popup=f"Cluster {cluster}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(athleteMap)

#save map
athleteMap.save("Athlete" + athlete + "Map.html")