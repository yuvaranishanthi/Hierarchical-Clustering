import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import joblib
import os
from scipy.spatial.distance import cdist

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train hierarchical clustering
model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
cluster_labels = model.fit_predict(X_scaled)

# Compute cluster centroids
centroids = []
for cluster_id in range(model.n_clusters):
    points_in_cluster = X_scaled[cluster_labels == cluster_id]
    centroid = points_in_cluster.mean(axis=0)
    centroids.append(centroid)
centroids = np.array(centroids)

# Save model components
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(features, "model/features.pkl")
joblib.dump(centroids, "model/centroids.pkl")

print("✅ Hierarchical clustering model saved for real-time predictions.")


