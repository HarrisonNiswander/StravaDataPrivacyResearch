#Harrison Niswander
#Honor's Project | CS 593 Ethics of AI
#Faculty Advisor: Professor Rusert
#3-28-26

import pandas as pd
import ast
import numpy as np

#Define Athlete (by letter)
athlete = "A"

#import athlete data
df = pd.read_csv('Athlete_Data/Athlete' + athlete + '.csv', on_bad_lines='skip')

#obtain time
df['start_date_local'] = pd.to_datetime(df['start_date_local'], errors='coerce')
df['hour'] = df['start_date_local'].dt.hour
df['day_of_week'] = df['start_date_local'].dt.dayofweek

#convert time into cyclic features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Convert starting longitude and latitude into seperate columns (coords in decminal degrees format)
# Convert string → list
df['start_latlng'] = df['start_latlng'].apply(ast.literal_eval)

# Split into separate columns
df['start_lat'] = df['start_latlng'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
df['start_lng'] = df['start_latlng'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)

# Drop missing values
startCoord = df[['start_lat', 'start_lng']].dropna()

#import DBSCAN (clustering Algorithm)
from sklearn.cluster import DBSCAN
import numpy as np

startCord_rad = np.radians(startCoord)

# DBSCAN with haversine metric -> convert to radians so we can do distance
kms_per_radian = 6371.0088
epsilon = 0.25 / kms_per_radian      #Radius for Clustering

db = DBSCAN(
    eps=epsilon,
    min_samples=20,                 #min number of activities to be considered cluster
    algorithm='ball_tree',
    metric='haversine'
).fit(startCord_rad)

startCoord['cluster'] = db.labels_

#determine meaningful clusters
startCoord['cluster'].value_counts()

#find the center of the clusters
cluster_centers = startCoord.groupby('cluster')[['start_lat','start_lng']].mean()
print(cluster_centers)

# Merge time data back in after DBSCAN
df = df.loc[startCoord.index].copy()
df['cluster'] = startCoord['cluster']

# Remove noise from cluster (-1)
data = df[df['cluster'] != -1].copy()

#
# Train a Machine Learning Model and remove 10% of data from each cluster
#
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#time features
features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
X = data[features]
y = data['cluster']

#10% Iterative Removal = Stratified Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

all_results = []

#repeat train and prediction for each 10% fold removal
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities (THIS is what you want)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Store results
    fold_results = pd.DataFrame(y_proba, columns=model.classes_)
    fold_results['true_cluster'] = y_test.values

    all_results.append(fold_results)

    #rank predicted clusters by confidence
    probs = model.predict_proba(X_test)

    # Get ranked clusters
    ranked_clusters = np.argsort(-probs, axis=1)

    #map back to cluster labels
    cluster_labels = model.classes_

    top_k = 3

    for i in range(len(X_test)):
        ranked = cluster_labels[ranked_clusters[i][:top_k]]
        confidence = probs[i][ranked_clusters[i][:top_k]]

        #print(f"Top predictions: {list(zip(ranked, confidence))}")

#final results
final_results = pd.concat(all_results)
print(final_results)