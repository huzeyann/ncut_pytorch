# Nyström Ncut (Complexity)

!!! abstract "TL;DR"
    
    1. **Sub-sample**: Solve Ncut on a smaller sub-sampled graph, reduce complexity.
    2. **kNN Propagation**: Propagate the Ncut values from sampled nodes to not-sampled.
  

---


The Nyström Ncut approximation dramatically improves computational complexity from quadratic to **linear time complexity** O(N), making it feasible to process large-scale graphs with millions of nodes.


## How Nyström Ncut works

**An intuition**: Nystrom approximation solve the large-scale problem on a small sub-sampled set, then propagate the solution from the small set to the large set.


The approach involves three main steps: (1) sub-sampling a subset of nodes, (2) computing the Normalized Cut (Ncut) on the sampled nodes, and (3) propagating the eigenvectors from the sampled nodes to the unsampled nodes.

### Sub-sampling the Affinity Matrix


<div style="text-align: center;">
<img src="/images/methods_02a_nystrom_ncut_complexity/subsample.png" style="width:80%;">
</div>

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
 
### k-Nearest Neighbors (kNN) Propagation

Farthest Point Sampling (FPS) is used to select nodes that are evenly distributed across the data,
we use [fpsample](https://github.com/leonardodalinky/fpsample).

<div style="text-align: center;">
<img src="/images/methods_02a_nystrom_ncut_complexity/knn_propagate.png" style="width:80%;">
</div>

After computing the eigenvectors \( \mathbf{X}' \in \mathbb{R}^{n \times C} \) on the sub-sampled graph \( S = A + \left({D_{\text{r}}}^{-1} B'\right) \left(B' {D_{\text{c}}}^{-1}\right)^\top \in \mathbb{R}^{n \times n} \) using the Ncut formulation, the next step is to propagate these eigenvectors to the full graph. Let \( \mathbf{\tilde{X}} \in \mathbb{R}^{N \times C} \) be the approximated eigenvectors for the full graph, where \( N \) is the total number of nodes. The eigenvector approximation \( \mathbf{\tilde{X}}_i \) for each node \( i \leq N \) in the full graph is obtained by averaging the eigenvectors \( \mathbf{X}'_k \) of the top K-nearest neighbors from the subgraph:

\[
\mathcal{K}_i = \text{KNN}(\mathbf{A}_{*i}; n, K) = argmax_{k \leq n} \sum_{k=1}^{K} \mathbf{B}_{ki}
\]

\[
\mathbf{\tilde{X}}_i = \frac{1}{\sum_{k \in \mathcal{K}_i} \mathbf{B}_{ki}} \sum_{k \in \mathcal{K}_i} \mathbf{B}_{ki} \mathbf{X}'_k 
\]

Here, \( \text{KNN}(\mathbf{A}_{*i}; n, K) \) denotes the set of K-nearest neighbors from the full-graph node \( i \leq N \) to the sub-graph nodes \( k \leq n \). In other words, the eigenvectors of un-sampled nodes are assigned by weighted averaging top KNN eigenvectors from sampled nodes.

<!-- The propagation step is only when \(B \in \mathbb{R}^{n \times (N-n)} \) is computed (but not stored), recall that the full \( B \) is too large to store, our implementation divide the propagation into chunks, where each $q$ size chunk \(B_i \in \mathbb{R}^{n \times (q)} \) is feasible to store, each chunk \(B_i\) is discarded after the propagation. In practice, the KNN propagation is the major speed bottle-neck for NCUT on large-scale graph, but KNN propagation is easy to parallelize on GPU.  -->



<div style="display: flex; justify-content: space-between; gap: 20px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
  <a href="/methods/01_basic_ncut" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Previous</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">← Basic Ncut</span>
  </a>
  <a href="/methods" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; transition: all 0.2s;">
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">Back to Overview</span>
  </a>
  <a href="/methods/02b_nystrom_ncut_quality" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: flex-end; text-align: right; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Next</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">Nyström Ncut (Quality) →</span>
  </a>
</div>
