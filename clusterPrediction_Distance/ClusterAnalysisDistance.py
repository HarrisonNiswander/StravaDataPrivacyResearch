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
results_path = os.path.join(script_dir, 'clusterPred_DistanceData', f'distanceAthlete{athlete}.csv')

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