# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min

# Generate synthetic data with 4 clusters
X, y = make_blobs(n_samples=10000, centers=4, cluster_std=0.50, random_state=0)

def farthest_point_sampling(X, n_samples):
    # Start with a random point
    n_points = X.shape[0]
    selected_points = [np.random.choice(n_points)]
    
    # Iteratively select the farthest point from the set of selected points
    for _ in range(1, n_samples):
        _, distances = pairwise_distances_argmin_min(X, X[selected_points])
        selected_points.append(np.argmax(distances))
        
    return np.array(selected_points)

# Run Farthest Point Sampling to select 10 points
n_samples = 100
fps_indices = farthest_point_sampling(X, n_samples)
fps_points = X[fps_indices]

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray', marker='o', alpha=0.5, label='Original Points')
plt.scatter(fps_points[:, 0], fps_points[:, 1], c='red', marker='x', s=100, label='Sampled Points')
plt.title(f'Farthest Point Sampling ({n_samples} points)')
plt.legend()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

# Generate synthetic data with 4 clusters
X, y = make_blobs(n_samples=2000, centers=4, cluster_std=0.60, random_state=0)

def farthest_point_sampling(X, n_samples):
    # Start with a random point
    n_points = X.shape[0]
    selected_points = [np.random.choice(n_points)]
    
    # Iteratively select the farthest point from the set of selected points
    for _ in range(1, n_samples):
        _, distances = pairwise_distances_argmin_min(X, X[selected_points])
        selected_points.append(np.argmax(distances))
        
    return np.array(selected_points)

# Run Farthest Point Sampling to select 10 points
n_samples = 100
fps_indices = farthest_point_sampling(X, n_samples)
fps_points = X[fps_indices]

# Create adjacency matrix from Euclidean distances between sampled points
adj_matrix_sampled = pairwise_distances(fps_points)

# Normalize adjacency matrix for line thickness
max_distance = np.max(adj_matrix_sampled)
normalized_adj_matrix = adj_matrix_sampled / max_distance

# Plot the points and connections
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray', marker='o', alpha=0.5, label='Original Points')
plt.scatter(fps_points[:, 0], fps_points[:, 1], c='red', marker='x', s=100, label='Sampled Points')

# Draw connections between sampled points
for i in range(n_samples):
    for j in range(i + 1, n_samples):
        if i != j:
            plt.plot([fps_points[i, 0], fps_points[j, 0]], [fps_points[i, 1], fps_points[j, 1]],
                     'b-', linewidth=normalized_adj_matrix[i, j] * 1, alpha=0.3)

plt.title(f'Farthest Point Sampling ({n_samples} points) with Connections')
plt.legend()
plt.show()
# %%
