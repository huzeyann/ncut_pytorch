# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

# Generate synthetic data with 4 clusters
X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)

# Compute the similarity matrix using RBF (Gaussian) kernel
sigma = 1.0
W = rbf_kernel(X, gamma=1/(2*sigma**2))

# Degree matrix
D = np.diag(np.sum(W, axis=1))

# Unnormalized Laplacian
L = D - W

# Normalized Laplacian (symmetric)
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
L_sym = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)

# Compute the eigenvalues and eigenvectors of the normalized Laplacian
eigvals, eigvecs = eigh(L_sym)

# Plot the first few eigenvectors
    
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
for i in range(4):
    ax[i//2, i%2].scatter(X[:, 0], X[:, 1], c=eigvecs[:, i], s=50, cmap='coolwarm')
    ax[i//2, i%2].set_title(f"Eigenvector {i+1} - Dividing the Graph")
plt.tight_layout()
plt.show()

k = 4
# Use the first k eigenvectors to form a matrix U
U = eigvecs[:, :k]

# Normalize U
U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)

# Perform k-means clustering on U_norm
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(U_norm)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title("Spectral Clustering: Clusters Found")
plt.show()
# %%
