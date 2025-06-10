import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score

df = pd.read_csv('watson_healthcare_modified.csv')

columns_to_drop = ['EmployeeID', 'EmployeeCount', 'Over18', 'StandardHours']
df = df.drop(columns=columns_to_drop)

y = df['Attrition'].map({'Yes': 1, 'No': 0})
df = df.drop(columns=['Attrition'])

categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

metrics = {
    'Silhouette Score': silhouette_score(X_scaled, clusters),
    'Adjusted Rand Index': adjusted_rand_score(y, clusters),
    'Normalized Mutual Info': normalized_mutual_info_score(y, clusters),
    'Homogeneity Score': homogeneity_score(y, clusters),
    'Completeness Score': completeness_score(y, clusters),
    'V-Measure Score': v_measure_score(y, clusters)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('K-means Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter1)

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.title('Actual Attrition Labels')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter2)

plt.tight_layout()
plt.show()
