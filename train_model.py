import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    df = pd.read_csv('Crop_recommendation.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'Crop_recommendation.csv' not found. Creating synthetic data for demonstration...")
    np.random.seed(42)
    df = pd.DataFrame({
        'N': np.random.randint(0, 140, 2200),
        'P': np.random.randint(5, 145, 2200),
        'K': np.random.randint(5, 205, 2200),
        'temperature': np.random.uniform(8, 45, 2200),
        'humidity': np.random.uniform(14, 100, 2200),
        'ph': np.random.uniform(3.5, 10, 2200),
        'rainfall': np.random.uniform(20, 300, 2200),
        'label': ['rice'] * 2200 # Dummy label column
    })
X = df.drop(columns=['label'])
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())
    print("Missing values handled with column means.")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('elbow_plot.png')
print("Elbow plot saved as 'elbow_plot.png'. Review this to verify K.")
optimal_k = 4 
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_transform(X_scaled)
df['Cluster'] = kmeans.labels_
score = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score for K={optimal_k}: {score:.4f}")
plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 6], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.title('Farmer Clusters (N vs Rainfall)')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[6])
plt.colorbar(label='Cluster')
plt.savefig('cluster_visualization.png')
print("Cluster visualization saved as 'cluster_visualization.png'.")
print("\nCluster Means Profile:")
print(df.drop(columns=['label']).groupby('Cluster').mean())
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("\nModel and Scaler saved successfully as 'kmeans_model.pkl' and 'scaler.pkl'.")