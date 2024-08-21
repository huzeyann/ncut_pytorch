# How NCUT works

## Normalized Cut and Spectral Clustering

Spectral clustering is a powerful technique for clustering data based on the eigenvectors (spectrum) of a similarity matrix derived from the data. The Normalized Cuts algorithm aims to partition a graph into subgraphs while minimizing the graph cut value.

### 1.1 The Basic Idea
Spectral clustering works by embedding the data points into a lower-dimensional space using the eigenvectors of a Laplacian matrix derived from the data's similarity graph. The data is then clustered in this new space.

### 1.2 The Graph Laplacian
Given a set of data points, spectral clustering first constructs a similarity graph. Each node represents a data point, and edges represent the similarity between data points. The similarity graph can be represented by an adjacency matrix \( W \), where each element \( W_{ij} \) represents the similarity between data points \( i \) and \( j \).

The degree matrix \( D \) is a diagonal matrix where each element \( D_{ii} \) is the sum of the similarities of node \( i \) to all other nodes.

The unnormalized graph Laplacian is defined as:

\[
L = D - W
\]

The normalized graph Laplacian has two common forms:

1. Symmetric normalized Laplacian:
 
\[
L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}
\]

2. Random walk normalized Laplacian:

\[
L_{\text{rw}} = D^{-1} L = I - D^{-1} W
\]


### 2.1 Normalized Cuts 
Normalized Cuts (Ncut) is a method for partitioning a graph into disjoint subsets, aiming to minimize the total edge weight between the subsets relative to the total edge weight within each subset. The Ncut criterion is particularly useful for ensuring balanced partitioning, which prevents trivial solutions where one cluster might be significantly smaller than the other.

The normalized cut criterion is defined as:

\[
\text{Ncut}(A, B) = \left(\frac{\text{cut}(A, B)}{\text{assoc}(A, V)}\right) + \left(\frac{\text{cut}(A, B)}{\text{assoc}(B, V)}\right)
\]

Where:

- \( A \) and \( B \) are two disjoint subsets (clusters) of the graph.

- \( \text{cut}(A, B) \) is the sum of the weights of the edges between the subsets \( A \) and \( B \):

\[
\text{cut}(A, B) = \sum_{i \in A, j \in B} W_{ij}
\]

- \( \text{assoc}(A, V) \) is the sum of the weights of all edges attached to nodes in subset \( A \) (similar definition for \( B \)):

\[
\text{assoc}(A, V) = \sum_{i \in A, j \in V} W_{ij}
\]

The goal is to find a partition that minimizes the Ncut value, which balances minimizing the inter-cluster connection (cut) with maintaining strong intra-cluster connections (association).

### 2.2 Solving the Normalized Cut Using Eigenvectors

To solve the Ncut problem, we reformulate it as a problem of finding the eigenvectors of a normalized graph Laplacian. Here’s how this works:

#### Mathematical Formulation

Let \( \mathbf{x} \) be an indicator vector such that:

\[
x_i = 
\begin{cases} 
1 & \text{if node } i \in A \\
-1 & \text{if node } i \in B 
\end{cases}
\]

The cut value \( \text{cut}(A, B) \) can be rewritten in terms of \( \mathbf{x} \) as:

\[
\text{cut}(A, B) = \frac{1}{4} \sum_{i,j} W_{ij} (x_i - x_j)^2 = \frac{1}{4} \mathbf{x}^\top L \mathbf{x}
\]

Where \( L = D - W \) is the unnormalized graph Laplacian.

The association \( \text{assoc}(A, V) \) is given by:

\[
\text{assoc}(A, V) = \mathbf{x}^\top D \mathbf{1}_A
\]

Here, \( \mathbf{1}_A \) is a vector that is 1 for elements in \( A \) and 0 elsewhere.

The Ncut problem can thus be rewritten as:

\[
\text{Ncut}(A, B) = \frac{\mathbf{x}^\top L \mathbf{x}}{\mathbf{x}^\top D \mathbf{x}}
\]

Minimizing this directly is NP-hard. However, it can be relaxed into a generalized eigenvalue problem. By relaxing the constraint that \( x_i \) takes only discrete values (1 or -1), we allow \( \mathbf{x} \) to take real values, and the problem becomes finding the eigenvector corresponding to the second smallest eigenvalue of the generalized eigenvalue problem:

\[
L \mathbf{y} = \lambda D \mathbf{y}
\]

This is equivalent to finding the eigenvectors of the normalized Laplacian:

\[
L_{\text{sym}} \mathbf{y} = \lambda \mathbf{y}
\]

Where \( L_{\text{sym}} = D^{-1/2} L D^{-1/2} \) is the symmetric normalized Laplacian.


Since \(Y Y^T = I \), Normalized Cuts can also be solved by eigenvectors of \(D^{-1/2} W D^{-1/2} \).

\[
I - D^{-1/2} W D^{-1/2} = Y \Lambda Y^T
\]

\[
D^{-1/2} W D^{-1/2} = Y (1 - \Lambda) Y^T
\]

#### Eigenvectors and Clustering

The eigenvector corresponding to the second smallest eigenvalue (also called the Fiedler vector) provides a real-valued solution to the relaxed Ncut problem. Sorting the nodes based on the values in this eigenvector and choosing a threshold to split them yields an approximate solution to the Ncut problem.

Each subsequent eigenvector can be used to further partition the graph into subclusters. The process is as follows:

1. Compute the symmetric normalized Laplacian \( L_{\text{sym}} \).
2. Compute the eigenvectors \( \mathbf{y}_1, \mathbf{y}_2, \dots, \mathbf{y}_k \) corresponding to the smallest \( k \) eigenvalues.
3. Use \( \mathbf{y}_2 \) (the Fiedler vector) to divide the graph into two clusters.
4. For multi-way clustering, use the next eigenvectors \( \mathbf{y}_3, \mathbf{y}_4, \dots \) to further subdivide these clusters.

##### Proof of Minimization

The eigenvector approach is derived from the relaxation of the original NP-hard problem. By solving the generalized eigenvalue problem:

\[
L \mathbf{y} = \lambda D \mathbf{y}
\]

We are effectively minimizing the Rayleigh quotient:

\[
\frac{\mathbf{y}^\top L \mathbf{y}}{\mathbf{y}^\top D \mathbf{y}}
\]

This quotient represents a trade-off between minimizing the cut (numerator) and balancing the partition (denominator), directly reflecting the Ncut objective.

##### Example: Eigenvector Visualization

Let's visualize how the eigenvectors divide the graph into clusters using our toy example.

```python
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
k = 4  # number of clusters
for i in range(1, k+1):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=eigvecs[:, i-1], s=50, cmap='coolwarm')
    plt.title(f"Eigenvector {i} - Dividing the Graph")
    plt.colorbar()
    plt.show()

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
```

<div style="text-align: center;">
<img src="../images/dots_eig.jpg.png" style="width:100%;">
</div>

- Each eigenvector reveals different partitions of the graph, showing how the spectral clustering process divides the dataset into meaningful subclusters.

- The first eigenvector generally corresponds to the global structure (often related to the connected components of the graph).

<div style="text-align: center;">
<img src="../images/dots_kmeans.png" style="width:60%;">
</div>

- Subsequent eigenvectors further refine the clustering by splitting the data into more detailed subclusters.

### Summary
The Normalized Cuts algorithm aims to partition a graph into subgraphs while minimizing the graph cut value. It embeds the data points into a lower-dimensional space using the eigenvectors of a Laplacian matrix derived from the data's similarity graph.

Eigenvectors are:

1) soft-cluster assignments, eigenvector in [-1, 1], -1 or 1 are extreme assignments, 0 means assigned to neither cluster. 

2) $n$ eigenvectors can hierarchically divide into $2^n$ sub-graphs.

------

<!-- ## Nystrom-like Approximation

Nystrom approximation aims to solve large-scale graph cuts. 

it 1) sub-sample a set of nodes. 2) Compute Ncut on the sampled nodes. 3) propagate the eigenvectors from sampled nodes to un-sampled nodes

Partition the affinity matrix W as

W =[ A B
B^T C]

with A in R(nxn), B \in R(N-n)xn, and C \in R(N-n)x(N-n). Here,
A represents the subblock of weights among the random
samples, B contains the weights from the random samples
to the rest of the pixels, and C contains the weights between
all of the remaining pixels. In the case of interest, n << N, so
C is huge. 

### Sub-sampling

A good sampling strategy is critical for a good approximation. 

Farthest point sampling is used, it ensure...

In practice FPS is expensive for high-dimensional input (model features), we use PCA to reduce the feature dimension.

### Indirect Connection

Not sampled nodes could be connecting two clusters, ignoring them will alter the cluster assignments

The original Nystrom Methods use \( S = A + A(-0.5) B B^T A^(-0.5) \) and solve eigenvector on S, this way B B^T will adds the indirect connection

we solve eigenvector on \( S = A + (1/D_row B) (1/D_col B'^T)  \), where D_row and D_col is row and column sum on B, (1/D_row B) (1/D_col B'^T) adds the indirect random walk probability

In practice, B in R(N-n)xn, storing and computing could be overwhelmingly expensive, we use PCA to reduce input number of nodes before computing B. Given feature size D, n sampled nodes F_A \in R(nxD), N-n un-sampled nodes F_B \in R((N-n)xD), F_B' \in R(mxD) is the PCA-ed version of F_B, where m << N-n, resulting B in R(m)xn greatly reduce the computation bottleneck.



### KNN Propagation 

Our Nystrom-like Approximation first solves the eigenvector $\bm{X}' \in \mathbb{R}^{m \times C}$ on a sub-sampled graph $\bm{A}' \in \mathbb{R}^{m \times m}$ using \Cref{eq:ncut}, then propagates the eigenvector from the sub-graph $m$ nodes to the full-graph $M$ nodes. Let $\bm{\Tilde{X}} \in \mathbb{R}^{M \times C}$ be the approximation $\bm{\Tilde{X}} \approx \bm{X}$. The eigenvector approximation $\bm{\Tilde{X}}_i$ of full-graph node $i \leq M$ is assigned by averaging the top K-nearest neighbors' eigenvector $\bm{X}'_k$ from the sub-graph nodes $k \leq m$:
\begin{equation}
\begin{aligned}
    \mathcal{K}_i &= KNN(\bm{A}_{*i}; m, K) = \argmax_{k \leq m} \sum_{k=1}^{K} \bm{A}_{ki}  \\
    % \bm{X}_i &= \frac{1}{|\mathcal{K}_i|} \sum_{k\in \mathcal{K}_i} \bm{A}_{ki} \bm{X}'_k 
    \bm{\Tilde{X}}_i &= \frac{1}{\sum_{k\in \mathcal{K}_i} \bm{A}_{ki}} \sum_{k\in \mathcal{K}_i} \bm{A}_{ki} \bm{X}'_k 
\end{aligned}
\end{equation}
where $KNN(\bm{A}_{*i}; m, K)$ denotes KNN from full-graph node $i \leq M$ to sub-graph nodes $k \leq m$.  -->


## Nystrom-like Approximation

The Nystrom-like approximation is our new technique designed to address the computational challenges of solving large-scale graph cuts. The approach involves three main steps: (1) sub-sampling a subset of nodes, (2) computing the Normalized Cut (Ncut) on the sampled nodes, and (3) propagating the eigenvectors from the sampled nodes to the unsampled nodes.

### Partitioning the Affinity Matrix

The affinity matrix \( W \) is partitioned as follows:

\[
W = \begin{pmatrix}
A & B \\
B^\top & C
\end{pmatrix}
\]

where:

- \( A \in \mathbb{R}^{n \times n} \) represents the submatrix of weights among the sampled nodes,

- \( B \in \mathbb{R}^{(N-n) \times n} \) contains the weights between the sampled nodes and the remaining unsampled nodes,

- \( C \in \mathbb{R}^{(N-n) \times (N-n)} \) represents the weights between the unsampled nodes.

In practical scenarios, \( n \) (the number of sampled nodes) is much smaller than \( N \) (the total number of nodes), making \( C \) a very large matrix that is computationally expensive to handle. \( B \) is also expensive to store and compute once sampled nodes \( n \) gets large.
 
### Sub-sampling

An effective sampling strategy is crucial for obtaining a good approximation of the Ncut. Farthest Point Sampling (FPS) is used because it ensures that the sampled nodes are well-distributed across the graph. However, FPS can be computationally expensive, especially for high-dimensional input data (e.g., model features). To mitigate this, Principal Component Analysis (PCA) is used to reduce the dimensionality of the features before applying FPS. We use a QuickFPS algorithm developed by 

- Reference: QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Cloud, Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng, 2023

### Accounting for Indirect Connections

Unsampled nodes can act as bridges between clusters, and ignoring these connections could lead to inaccurate cluster assignments. To address this, the original Nystrom method [2] approximates the eigenvectors by solving the following matrix \( S \):

\[
S = A + A^{-1/2} B B^\top A^{-1/2}
\]

Here, the term \( B B^\top \) accounts for indirect connections between the sampled nodes by considering the influence of the unsampled nodes.

In our method, we refine this approximation by solving for eigenvectors on the matrix \( S \) given by:

\[
S = A + \left({D_{\text{r}}}^{-1} B\right) \left(B {D_{\text{c}}}^{-1}\right)^\top
\]

where \( D_{\text{r}} \) and \( D_{\text{c}} \) are the row and column sums of matrix \( B \), respectively. \( \left({D_{\text{r}}}^{-1} B\right) \left(B {D_{\text{c}}}^{-1}\right)^\top \) is the indirect random walk probabilities.

Given that \( B \) has dimensions \( (N-n) \times n \), directly storing and computing with \( B \) can be prohibitively expensive. To overcome this, we reduce the number of unsampled nodes by applying PCA. If the feature size is \( D \), let \( F_A \in \mathbb{R}^{n \times D} \) be the feature matrix for the sampled nodes, and \( F_B \in \mathbb{R}^{(N-n) \times D} \) be the feature matrix for the unsampled nodes. After applying PCA, the reduced feature matrix \( F_B' \in \mathbb{R}^{m \times D} \) represents the unsampled nodes, where \( m \ll (N-n) \). Thus, \( B \) becomes a matrix of size \( m \times n \), significantly reducing computational overhead.

### K-Nearest Neighbors (KNN) Propagation

After computing the eigenvectors \( \mathbf{X}' \in \mathbb{R}^{n \times C} \) on the sub-sampled graph \( \mathbf{S} \in \mathbb{R}^{n \times n} \) using the Ncut formulation, the next step is to propagate these eigenvectors to the full graph. Let \( \mathbf{\tilde{X}} \in \mathbb{R}^{N \times C} \) be the approximated eigenvectors for the full graph, where \( N \) is the total number of nodes. The eigenvector approximation \( \mathbf{\tilde{X}}_i \) for each node \( i \leq N \) in the full graph is obtained by averaging the eigenvectors \( \mathbf{X}'_k \) of the top K-nearest neighbors from the subgraph:

\[
\mathcal{K}_i = \text{KNN}(\mathbf{A}_{*i}; n, K) = argmax_{k \leq n} \sum_{k=1}^{K} \mathbf{A}_{ki}
\]

\[
\mathbf{\tilde{X}}_i = \frac{1}{\sum_{k \in \mathcal{K}_i} \mathbf{A}_{ki}} \sum_{k \in \mathcal{K}_i} \mathbf{A}_{ki} \mathbf{X}'_k 
\]

Here, \( \text{KNN}(\mathbf{A}_{*i}; n, K) \) denotes the set of K-nearest neighbors from the full-graph node \( i \leq N \) to the sub-graph nodes \( k \leq n \). In other words, the eigenvectors of un-sampled nodes are assigned by weighted averaging top KNN eigenvectors from sampled nodes.

---



> paper in prep, Yang 2024
>
> [1] AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*, 2024
>
> [2] Spectral Grouping Using the Nystrom Method, Charless Fowlkes, Serge Belongie, Fan Chung, and Jitendra Malik, 2004
> 
> [3] Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
> 