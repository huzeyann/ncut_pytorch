# How NCUT works

## Normalized Cuts and Spectral Clustering

Spectral clustering is a powerful technique for clustering data based on the eigenvectors (spectrum) of a similarity matrix derived from the data. The Normalized Cuts algorithm aims to partition a graph into subgraphs while minimizing the graph cut value.

<div style="text-align: center;">
<img src="../images/graph_cut.png" style="width:100%;">
</div>
Image taken from [Spectral Clustering: Step-by-step derivation of the spectral clustering algorithm including an implementation in Python](https://medium.com/@roiyeho/spectral-clustering-50aee862d300)


---

### The Basic Idea
Spectral clustering works by embedding the data points $F \in \mathbb{R}^{N \times D}$ into a lower-dimensional space using the eigenvectors of a Laplacian matrix derived from the data's similarity graph $W \in \mathbb{R}^{N \times N}$. The data is then clustered in this new space embedded by $k$ eigenvectors $\mathbf{x} \in \mathbb{R}^{N \times k}$.

### The Graph Laplacian
Given a set of data points, spectral clustering first constructs a similarity graph. Each node represents a data point, and edges represent the similarity between data points. The similarity graph can be represented by an adjacency matrix \( W \), where each element \( W_{ij} \) represents the similarity between data points \( i \) and \( j \).


Take cosine distance for example, let $F \in \mathbb{R}^{N \times D}$ be the input features (from a backbone image model), $N$ is number of pixels, $D$ is feature dimension. Feature vectors $f_i, f_j \in \mathbb{R}^D$, the cosine similarity between $f_i$ and $f_j$ is defined as:

$$W_{ij} = \text{cosine}(f_i, f_j) = \frac{f_i \cdot f_j}{|f_i| |f_j|}$$

where $|f|$ denotes the Euclidean norm of a vector $f$.

In matrix form, this can be written as:

$$W = \text{cosine}(F, F) = \frac{F F^\top}{\text{diag}(F F^\top)^{1/2} , \text{diag}(F F^\top)^{1/2}}$$

where $F F^\top \in \mathbb{R}^{N \times N}$ is the matrix of pairwise dot products between the feature vectors, and $\text{diag}(\cdot)$ extracts the diagonal elements of a square matrix.
The resulting matrix $W = \text{cosine}(F, F)$ is an $\mathbb{R}^{N \times N}$ matrix, where each element $(i, j)$ represents the cosine similarity between the $i$-th and $j$-th feature vectors.


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


### Normalized Cuts 
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


### Solving the Normalized Cut Using Eigenvectors

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

To make it a simple eigenvalue solving, that can be solved by most eigenvector solvers. 
Move $D$ to the left side of the equation. this is equivalent to finding the eigenvectors of the normalized Laplacian:

\[
D^{-1/2} L D^{-1/2} \mathbf{y} = \lambda \mathbf{y}
\]


#### Proof of Minimization

The eigenvector approach is derived from the relaxation of the original NP-hard problem. By solving the generalized eigenvalue problem:

\[
L \mathbf{y} = \lambda D \mathbf{y}
\]

We are effectively minimizing the Rayleigh quotient:

\[
\frac{\mathbf{y}^\top L \mathbf{y}}{\mathbf{y}^\top D \mathbf{y}} = \frac{\mathbf{y}^\top D^{-1/2} L D^{-1/2} \mathbf{y}}{\mathbf{y}^\top  \mathbf{y}}
\]

The Rayleigh quotient directly reflecting the Ncut objective.

\[
\text{Ncut}(A, B) = \frac{\mathbf{x}^\top L \mathbf{x}}{\mathbf{x}^\top D \mathbf{x}} = \frac{\mathbf{x}^\top D^{-1/2} L D^{-1/2} \mathbf{x}}{\mathbf{x}^\top \mathbf{x}}
\]

thus, solving eigenvector $\mathbf{y}$ of $D^{-1/2} L D^{-1/2} = \lambda \mathbf{y}$ is equal to solving the graph cut solution $\mathbf{x}$:

\[ 
    \mathbf{x} = \mathbf{y} 
\]


#### Laplacian vs Affinity

Since \(Y Y^T = I \), Normalized Cuts can also be solved by eigenvectors of \(D^{-1/2} W D^{-1/2} \), instead of \(D^{-1/2} L D^{-1/2} \), where $L = D - W$.

$$\begin{aligned}
D^{-1/2} (D - W) D^{-1/2} &= Y \Lambda Y^T \\
I - D^{-1/2} W D^{-1/2} &= Y \Lambda Y^T \\
D^{-1/2} W D^{-1/2} &= Y (I - \Lambda) Y^T
\end{aligned}$$

#### Eigenvectors and Clustering

The eigenvector corresponding to the second smallest eigenvalue (also called the Fiedler vector) provides a real-valued solution to the relaxed Ncut problem. Sorting the nodes based on the values in this eigenvector and choosing a threshold to split them yields an approximate solution to the Ncut problem.

Each subsequent eigenvector can be used to further partition the graph into subclusters. The process is as follows:

1. Compute the symmetric normalized Laplacian \( L_{\text{sym}} \).
2. Compute the eigenvectors \( \mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_k \) corresponding to the smallest \( k \) eigenvalues.
3. Use \( \mathbf{x}_2 \) (the Fiedler vector) to divide the graph into two clusters.
4. For multi-way clustering, use the next eigenvectors \( \mathbf{x}_3, \mathbf{x}_4, \dots \) to further hierarchically subdivide these clusters.

##### Eigenvector Solver

Solving the full eigenvector $\mathbf{x} \in \mathbb{R}^{N \times N}$ is computational expensive, methods has been developed to solve top k eigenvectors $\mathbf{x} \in \mathbb{R}^{N \times k}$ in linearly complexity scaling. In particular, we use [svd_lowrank](https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html).

- Reference: 

    Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions, 2009.



#### Example: Eigenvector Visualization

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

- The 1-st eigenvector generally corresponds to the global structure (often related to the connected components of the graph).

- The 2-nd eigenvector splits the graph into 2 clusters.

- Subsequent eigenvectors further hierarchically splits the data into more detailed sub-clusters.


<div style="text-align: center;">
<img src="../images/dots_kmeans.png" style="width:60%;">
</div>


### Summary
The Normalized Cuts algorithm aims to partition a graph into subgraphs while minimizing the graph cut value. It embeds the data points into a lower-dimensional space using the eigenvectors of a Laplacian matrix derived from the data's similarity graph.

Eigenvectors are:

1) soft-cluster assignments, eigenvector in [-1, 1], -1 or 1 are extreme assignments, 0 means assigned to neither cluster. 

2) $n$ eigenvectors can hierarchically divide into $2^n$ sub-graphs.

------



## Nystrom-like Approximation

The Nystrom-like approximation is our new technique designed to address the computational challenges of solving large-scale graph cuts. 

**An intuition**: Nystrom approximation solve the large-scale problem on a small sub-sampled set, then propagate the solution from the small set to the large set.

The approach involves three main steps: (1) sub-sampling a subset of nodes, (2) computing the Normalized Cut (Ncut) on the sampled nodes, and (3) propagating the eigenvectors from the sampled nodes to the unsampled nodes.

### Sub-sampling the Affinity Matrix

The affinity matrix \( W \) is partitioned as follows:

\[
W = \begin{pmatrix}
A & B \\
B^\top & C
\end{pmatrix}
\]

where:

- \( A \in \mathbb{R}^{n \times n} \) is the submatrix of weights among the sampled nodes,

- \( B \in \mathbb{R}^{(N-n) \times n} \) is the weights between the sampled nodes and the remaining unsampled nodes,

- \( C \in \mathbb{R}^{(N-n) \times (N-n)} \) is the weights between the unsampled nodes.

**Computation loads**: \( C \) is a very large matrix that is computationally expensive to handle. \( B \) is also expensive to store and compute once sampled nodes \( n \) gets large, although \( n \) (the number of sampled nodes) is much smaller than \( N \) (the total number of nodes), a better quality of  Nystrom approximation requires \( n \) to be larger, thus makes \( B \) a large matrix.
 
### A Good Sampling Strategy: FPS

An effective sampling strategy is crucial for obtaining a good approximation of the Ncut. Farthest Point Sampling (FPS) is used because it ensures that the sampled nodes are well-distributed across the graph. 
We use a tree-based QuickFPS algorithm developed in 

- Reference: 

    QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Cloud, Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng, 2023

As a side note, for high-dimensional input model features $F \in \mathbb{R}^{N \times D}$ where $D$ is feature dimension, FPS can be computationally expensive for large $D$. To mitigate this, Principal Component Analysis (PCA) is used to reduce the dimensionality of the features to $F' \in \mathbb{R}^{N \times 5}$. PCA is only applied in the FPS sampling step, but not to the affinity graph $W$.

### Accounting for Indirect Connections

Unsampled nodes can act as bridges between clusters, and ignoring these connections could lead to inaccurate cluster assignments. To address this, the original Nystrom method [2] approximates the eigenvectors by solving the following matrix \( S \):

\[
S = A + A^{-1/2} B B^\top A^{-1/2}
\]

Here, the term \( A^{-1/2} B B^\top A^{-1/2} \in \mathbb{R}^{n \times n} \) accounts for indirect connections between the sampled nodes by considering the influence of the unsampled nodes.

In our method, we solve for eigenvectors on the matrix \( S \) given by:

\[
S = A + \left({D_{\text{r}}}^{-1} B\right) \left(B {D_{\text{c}}}^{-1}\right)^\top
\]

where \( D_{\text{r}} \) and \( D_{\text{c}} \) are the row and column sums of matrix \( B \), respectively. 

the term \( \left({D_{\text{r}}}^{-1} B\right) \left(B {D_{\text{c}}}^{-1}\right)^\top \in \mathbb{R}^{n \times n}\) is the indirect random walk probabilities.

Given that \( B \) has dimensions \( (N-n) \times n \), directly storing and computing with \( B \) can be prohibitively expensive. To overcome this, we reduce the number of unsampled nodes by applying PCA. If the feature size is \( D \), let \( F_A \in \mathbb{R}^{n \times D} \) be the feature matrix for the sampled nodes, and \( F_B \in \mathbb{R}^{(N-n) \times D} \) be the feature matrix for the unsampled nodes. After applying PCA, $m$ is the PCA-ed dimension, the reduced feature matrix \( F_B' \in \mathbb{R}^{m \times D} \) represents the unsampled nodes, where \( m \ll (N-n) \). Thus, \( B' = d(F_A, F_{B'}) \) becomes a matrix of size \( m \times n \), thus significantly reduce computation loads.

### K-Nearest Neighbors (KNN) Propagation

After computing the eigenvectors \( \mathbf{X}' \in \mathbb{R}^{n \times C} \) on the sub-sampled graph \( S = A + \left({D_{\text{r}}}^{-1} B'\right) \left(B' {D_{\text{c}}}^{-1}\right)^\top \in \mathbb{R}^{n \times n} \) using the Ncut formulation, the next step is to propagate these eigenvectors to the full graph. Let \( \mathbf{\tilde{X}} \in \mathbb{R}^{N \times C} \) be the approximated eigenvectors for the full graph, where \( N \) is the total number of nodes. The eigenvector approximation \( \mathbf{\tilde{X}}_i \) for each node \( i \leq N \) in the full graph is obtained by averaging the eigenvectors \( \mathbf{X}'_k \) of the top K-nearest neighbors from the subgraph:

\[
\mathcal{K}_i = \text{KNN}(\mathbf{A}_{*i}; n, K) = argmax_{k \leq n} \sum_{k=1}^{K} \mathbf{B}_{ki}
\]

\[
\mathbf{\tilde{X}}_i = \frac{1}{\sum_{k \in \mathcal{K}_i} \mathbf{B}_{ki}} \sum_{k \in \mathcal{K}_i} \mathbf{B}_{ki} \mathbf{X}'_k 
\]

Here, \( \text{KNN}(\mathbf{A}_{*i}; n, K) \) denotes the set of K-nearest neighbors from the full-graph node \( i \leq N \) to the sub-graph nodes \( k \leq n \). In other words, the eigenvectors of un-sampled nodes are assigned by weighted averaging top KNN eigenvectors from sampled nodes.

The propagation step is only when \(B \in \mathbb{R}^{n \times (N-n)} \) is computed (but not stored), recall that the full \( B \) is too large to store, our implementation divide the propagation into chunks, where each $q$ size chunk \(B_i \in \mathbb{R}^{n \times (q)} \) is feasible to store, each chunk \(B_i\) is discarded after the propagation. In practice, the KNN propagation is the major speed bottle-neck for NCUT on large-scale graph, but it's easy to parallelize on GPU. 

```py
# GPU speeds up the KNN propagation
eigvectors, eigvalues = NCUT(num_eig=100, device='cuda:0').fit_transform(data)
```

---



> paper in prep, Yang 2024
>
> [1] AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*, 2024
>
> [2] Spectral Grouping Using the Nystrom Method, Charless Fowlkes, Serge Belongie, Fan Chung, and Jitendra Malik, 2004
> 
> [3] Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
> 