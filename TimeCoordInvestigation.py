# #Harrison Niswander
# #Honor's Project | CS 593 Ethics of AI
# #Faculty Advisor: Professor Rusert
# #3-28-26

# import numpy as np

# def safe_parse_coords(x):
#     try:
#         if pd.isna(x):
#             return [np.nan, np.nan]
        
#         coords = ast.literal_eval(x)
        
#         if isinstance(coords, (list, tuple)) and len(coords) >= 2:
#             return [float(coords[0]), float(coords[1])]
#         else:
#             return [np.nan, np.nan]
#     except:
#         return [np.nan, np.nan]

# import pandas as pd
# import ast

# #Define Athlete (by letter)
# athlete = "A"

# #import athlete data
# df = pd.read_csv('Athlete_Data/Athlete' + athlete + '.csv', on_bad_lines='skip')

# # parse latitude and longitude
# df[['start_lat', 'start_lng']] = df['start_latlng'].apply(
#     lambda x: pd.Series(safe_parse_coords(x))
# )

# df['start_lat'] = pd.to_numeric(df['start_lat'], errors='coerce')
# df['start_lng'] = pd.to_numeric(df['start_lng'], errors='coerce')

# # parse start time
# df['start_datetime'] = pd.to_datetime(df['start_date_local'], errors='coerce')
# df['start_timestamp'] = df['start_datetime'].astype('int64') // 10**9
# df['time_hours'] = df['start_datetime'].dt.hour + df['start_datetime'].dt.minute / 60

# # clean data up
# df = df.dropna(subset=['start_lat', 'start_lng', 'start_date_local'])

# # Convert time to hours
# df['time_hours'] = df['start_date_local'] / 3600

# # Features
# features = df[['start_lat', 'start_lng', 'time_hours']]

# # Scale
# from sklearn.preprocessing import StandardScaler
# X = StandardScaler().fit_transform(features)

# # DBSCAN
# from sklearn.cluster import DBSCAN
# dbscan = DBSCAN(eps=0.25, min_samples=20)
# df['cluster'] = dbscan.fit_predict(X)

# # Rebuild dataset for mapping
# startCoord = df[['start_lat', 'start_lng', 'cluster']]

# #cluster summary
# cluster_summary = (
#     df[df['cluster'] != -1]
#     .groupby('cluster')
#     .agg(
#         lat=('start_lat', 'mean'),
#         lng=('start_lng', 'mean'),
#         start_time=('start_datetime', 'min'),
#         end_time=('start_datetime', 'max')
#     )
# )

# cluster_summary['label'] = (
#     cluster_summary['start_time'].dt.strftime('%H:%M') +
#     " – " +
#     cluster_summary['end_time'].dt.strftime('%H:%M')
# )

# #find the center of the clusters
# cluster_centers = startCoord.groupby('cluster')[['start_lat','start_lng']].mean()
# #print(cluster_centers)

# #convert back to readable time
# df['start_datetime'] = pd.to_datetime(df['start_date_local'], unit='s')

# #time labels for clusters
# cluster_time_labels = df[df['cluster'] != -1].groupby('cluster')['start_datetime'].agg(['min', 'max'])
# cluster_time_labels['label'] = cluster_time_labels.apply(
#     lambda row: f"{row['min'].strftime('%H:%M')}–{row['max'].strftime('%H:%M')}",
#     axis=1
# )
# #print(cluster_time_labels)

# df['cluster_label'] = df['cluster'].map(cluster_time_labels['label'])

# #cluster by rounded hour
# df['hour'] = df['start_datetime'].dt.hour

# cluster_hours = df[df['cluster'] != -1].groupby('cluster')['hour'].mean()
# #print(cluster_hours)

# #use Folium to create real map of clusters
# import folium

# #center map around data
# center_lat = startCoord['start_lat'].mean()
# center_lng = startCoord['start_lng'].mean()

# athleteMap = folium.Map(location=[center_lat, center_lng], zoom_start=12)

# #define colors for clusters
# import random

# uniqueClusters = startCoord['cluster'].unique()

# clusterColors = {
#     cluster: "#{:06x}".format(random.randint(0, 0xFFFFFF))
#     for cluster in uniqueClusters if cluster != -1
# }

# #plot points
# for _, row in df.iterrows():
#     cluster = row['cluster']

#     if cluster == -1:
#         color = 'gray'
#         popup_text = "Noise"
#     else:
#         color = clusterColors.get(cluster, 'blue')
#         popup_text = f"Cluster {cluster}<br>{cluster_summary.loc[cluster, 'label']}"

# #highlight where the center of clusters are at
# centers = startCoord[startCoord['cluster'] != -1].groupby('cluster')[['start_lat','start_lng']].mean()

# for cluster, row in centers.iterrows():
#     folium.Marker(
#         location=[row['start_lat'], row['start_lng']],
#         popup=f"Cluster {cluster}<br>{row['label']}",
#         icon=folium.Icon(color='red', icon='info-sign')
#     ).add_to(athleteMap)

# #add heat map
# # from folium.plugins import HeatMap
# # HeatMap(startCoord[['start_lat','start_lng']].values).add_to(athleteMap)

# #save map
# athleteMap.save("Athlete_TimeMap/Athlete" + athlete + "Map.html")
#
#
#
#
#
#
#
#



#Harrison Niswander
#Honor's Project | CS 593 Ethics of AI
#Faculty Advisor: Professor Rusert
#3-28-26

import pandas as pd
import ast

#Define Athlete (by letter)
athlete = "A"

#import athlete data
df = pd.read_csv('Athlete_Data/Athlete' + athlete + '.csv', on_bad_lines='skip')

#obtain time
df['start_date_local'] = pd.to_datetime(df['start_date_local'], errors='coerce')
df['hour'] = df['start_date_local'].dt.hour
df['day_of_week'] = df['start_date_local'].dt.dayofweek

# Convert starting longitude and latitude into seperate columns (coords in decminal degrees format)
# Convert string → list
df['start_latlng'] = df['start_latlng'].apply(ast.literal_eval)

# Split into separate columns
df['start_lat'] = df['start_latlng'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
df['start_lng'] = df['start_latlng'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)

# Drop missing values
startCoord = df[['start_lat', 'start_lng', 'hour']].dropna()

# Cyclical encodying for time
import numpy as np
startCoord['hour_sin'] = np.sin(2 * np.pi * startCoord['hour'] / 24)
startCoord['hour_cos'] = np.cos(2 * np.pi * startCoord['hour'] / 24)

features = startCoord[['start_lat', 'start_lng', 'hour_sin', 'hour_cos']]

#import DBSCAN (clustering Algorithm)
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

features = startCoord[['start_lat', 'start_lng', 'hour_sin', 'hour_cos']]
X_scaled = StandardScaler().fit_transform(features)

# DBSCAN with haversine metric -> convert to radians so we can do distance
kms_per_radian = 6371.0088
epsilon = 0.25 / kms_per_radian      #Radius for Clustering

db = DBSCAN(
    eps=0.6,
    min_samples=20
).fit(X_scaled)

startCoord['cluster'] = db.labels_

df.loc[startCoord.index, 'cluster'] = startCoord['cluster']

#determine meaningful clusters
#startCoord['cluster'].value_counts()

#find the center of the clusters
cluster_centers = startCoord.groupby('cluster')[['start_lat','start_lng']].mean()
print(cluster_centers)

#use Folium to create real map of clusters
import folium

#center map around data
center_lat = startCoord['start_lat'].mean()
center_lng = startCoord['start_lng'].mean()

athleteMap = folium.Map(location=[center_lat, center_lng], zoom_start=12)

#define colors for clusters
import random

uniqueClusters = startCoord['cluster'].unique()

clusterColors = {
    cluster: "#{:06x}".format(random.randint(0, 0xFFFFFF))
    for cluster in uniqueClusters if cluster != -1
}

#plot points
for _, row in df.dropna(subset=['cluster']).iterrows():
    cluster = row['cluster']

    if cluster == -1:
        color = 'gray'
    else:
        color = clusterColors[cluster]

    popup_text = f"""
    Cluster: {cluster}<br>
    Time: {row['start_date_local']}<br>
    Hour: {row['hour']}
    """

    folium.CircleMarker(
        location=[row['start_lat'], row['start_lng']],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=250)
    ).add_to(athleteMap)

#highlight where the center of clusters are at
centers = startCoord[startCoord['cluster'] != -1].groupby('cluster')[['start_lat','start_lng']].mean()

for cluster, row in centers.iterrows():
    folium.Marker(
        location=[row['start_lat'], row['start_lng']],
        popup=f"Cluster {cluster}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(athleteMap)

#summary of clusters
cluster_time_summary = df.groupby('cluster')['hour'].agg(['mean', 'min', 'max'])
print(cluster_time_summary)

#add heat map
# from folium.plugins import HeatMap
# HeatMap(startCoord[['start_lat','start_lng']].values).add_to(athleteMap)

#save map
athleteMap.save("Athlete_TimeMap/Athlete" + athlete + "MapTime.html")