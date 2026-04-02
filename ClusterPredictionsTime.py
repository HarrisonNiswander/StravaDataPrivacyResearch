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
print("Cluster Centers (Lat, Lng):\n")
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

#repeat train and prediction for each 10% fold removal
top_k = 5       
all_results = []
print("\nStarting Cross-Validation and Prediction...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    cluster_labels = model.classes_

    for i in range(len(X_test)):
        prob_row = probs[i]

        # Rank clusters by confidence
        ranked_idx = np.argsort(-prob_row)
        top_clusters = cluster_labels[ranked_idx[:top_k]]
        top_conf = prob_row[ranked_idx[:top_k]]

        result = {
            'fold': fold + 1,
            'true_cluster': y_test.iloc[i],
            'predicted_cluster': preds[i],
            'correct': preds[i] == y_test.iloc[i],
        }

        # Add top-k predictions
        for k in range(top_k):
            result[f'top{k+1}_cluster'] = top_clusters[k]
            result[f'top{k+1}_confidence'] = top_conf[k]

        #Rank of true cluster in predictions
        ranked_idx = np.argsort(-prob_row)
        ranked_clusters = cluster_labels[ranked_idx]

        true_rank = np.where(ranked_clusters == y_test.iloc[i])[0][0] + 1
        result['true_rank'] = true_rank


        all_results.append(result)

#save all results into dataframe
results_df = pd.DataFrame(all_results)

#save results to csv
results_df.to_csv('coordCluster_TimePredict/coordCluster_TimePredict_Athlete' + athlete + '.csv', index=False)

#print useful results
print("\n--------------------------------------------")
print("Prediction Results for Athlete " + athlete)
print("--------------------------------------------\n")

#overall accuracy
print("Overall Accuracy:", results_df['correct'].mean())

#rank accuracy
max_k = 9   # or number of clusters

rank_accuracies = {}

for k in range(1, max_k + 1):
    rank_accuracies[k] = (results_df['true_rank'] <= k).mean()

#print results
print("\nCluster Prediction Accuracy by Rank:")
for k, acc in rank_accuracies.items():
    print(f"Top-{k} Accuracy: {acc:.4f}")

#graph
import matplotlib.pyplot as plt

plt.plot(list(rank_accuracies.keys()), list(rank_accuracies.values()))
plt.xlabel("K (Rank)")
plt.ylabel("Accuracy (Hit Rate)")
plt.title("Top-K Cluster Prediction Accuracy")
plt.show()