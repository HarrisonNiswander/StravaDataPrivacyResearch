#Harrison Niswander
#Honor's Project | CS 593 Ethics of AI
#Faculty Advisor: Professor Rusert
#4-2-26

import pandas as pd
import os

#Define Athlete (by letter)
athlete = "A"

# Get script directory and construct path
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, 'clusterPred_StartTimeData', f'startTimeAthlete{athlete}.csv')

#import cluster data
results = pd.read_csv(results_path, on_bad_lines='skip')

#
# Determine which cluster locations were easier to predict based on feature
#

#cluster prediction accuracy ranked
cluster_accuracy = results.groupby('true_cluster')['correct'].mean()
print("Cluster Prediction Accuracy by Cluster:")
print(cluster_accuracy.sort_values(ascending=False))

#cluster prediction for top 5 predicted clusters
for k in range(1, 6):
    results[f'top{k}_hit'] = (
        results['true_cluster'] == results[f'top{k}_cluster']
    )

cluster_topk = {}

for k in range(1, 6):
    col_name = f'top{k}_correct'
    
    results[col_name] = False
    for j in range(1, k+1):
        results[col_name] |= (
            results['true_cluster'] == results[f'top{j}_cluster']
        )
    
    cluster_topk[k] = results.groupby('true_cluster')[col_name].mean()

#compile into table
summary = pd.DataFrame({
    'top1_acc': cluster_topk[1],
    'top2_acc': cluster_topk[2],
    'top3_acc': cluster_topk[3],
    'top4_acc': cluster_topk[4],
    'top5_acc': cluster_topk[5],
})

#add number of samples per cluster to summary
counts = results['true_cluster'].value_counts()
summary['num_samples'] = counts

#print summary table
print("\nCluster Prediction (Top 5) Accuracy by Cluster:\n")
print(summary.sort_values('top1_acc', ascending=False))

#
# Determine if certain day or time clusters are more predictable than others
#

# Analyze accuracy by hour of day
hourAccuracy = results.groupby('hour')['correct'].mean()
print("\nCluster Prediction Accuracy by Hour of Day:")
print(hourAccuracy.sort_values(ascending=False))

# Analyze accuracy by day of week
downAccuracy = results.groupby('day_of_week')['correct'].mean()

day_map = {
    0: 'Mon', 1: 'Tue', 2: 'Wed',
    3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}

downAccuracy.index = downAccuracy.index.map(day_map)
print("\nCluster Prediction Accuracy by Day of Week:")
print(downAccuracy.sort_values(ascending=False))

# Analyze top-3 accuracy by hour of day
results['top3_correct'] = False

for k in range(1, 4):
    results['top3_correct'] |= (
        results['true_cluster'] == results[f'top{k}_cluster']
    )

hour_top3 = results.groupby('hour')['top3_correct'].mean()

for k in range(1, 3):
    results['top2_correct'] |= (
        results['true_cluster'] == results[f'top{k}_cluster']
    )

hour_top2 = results.groupby('hour')['top2_correct'].mean()

#put in table
hour_summary = pd.DataFrame({
    'top1_acc': results.groupby('hour')['correct'].mean(),
    'top2_acc': results.groupby('hour')['top2_correct'].mean(),
    'top3_acc': results.groupby('hour')['top3_correct'].mean(),
    'count': results['hour'].value_counts()
})
print("\nTop 3 Cluster Prediction Accuracy by Hour of Day:")
print(hour_summary.sort_values('top1_acc', ascending=False))

# Table for accuracy for time of day and day of week
pivot = results.pivot_table(
    values='correct',
    index='hour',
    columns='day_of_week',
    aggfunc='mean'
)
print("\nAccuracy for Time of Day and Day of Week:")
print(pivot)
