
# NCUT Parameters

## num_eig

number of eigenvectors, more eigenvectors give more details on the segmentation, but is more eigenvectors is harder to visualize.

In NCUT, by math design (see [How NCUT Works](how_ncut_works.md)), i-th eigenvector give a optimal graph cut that divides the graph into 2^(i-1) clusters. E.g. `eigenvectors[:, 1]` divides the graph into 2 clusters, `eigenvectors[:, 1:4]` divides the graph into 8 clusters.

<div style="text-align: center;">
<img src="../images/parameters/n_eig=10_n_images=100.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/n_eig=20_n_images=100.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/n_eig=30_n_images=100.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/n_eig=40_n_images=100.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/n_eig=50_n_images=100.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/n_eig=100_n_images=100.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/n_eig=1000_n_images=100.png" style="width:100%;">
</div>

## knn

propagation smoothness: higher knn means smoother propagation, higher knn value will give smoother eigenvector outputs.

In Nystrom NCUT, KNN is used to propagate nystrom sampled nodes to not-sampled nodes, each not-sampled nodes only take the top K nearest neighbor to propagate eigenvectors.

<div style="text-align: center;">
<img src="../images/parameters/knn=1.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/knn=3.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/knn=10.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/knn=100.png" style="width:100%;">
</div>

## affinity_focal_gamma

sharpness for affinity, range is \(0, 1\]. Each element A_ij in affinity is transformed as 

```
A_ij = 1 - cos_similarity(node_i, node_j)
A_ij = exp(-(A_ij / affinity_focal_gamma))
```

lower `affinity_focal_gamma` means shaper affinity. 

This transform is inspired by [focal loss](https://paperswithcode.com/method/focal-loss). In graph cut methods, disconnected edges are more important than connected edges, adding edge between two already connected clusters will not greatly alter the resulting eigenvectors, however, adding edge between two disconnected clusters will greatly alter the eigenvectors. Lower the `affinity_focal_gamma` value will make less-connected edges fade away, the resulting eigenvector will be shaper.


<div style="text-align: center;">
<img src="../images/parameters/t=0.1.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/t=0.2.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/t=0.3.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/t=0.4.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/t=0.5.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/t=0.6.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/t=0.7.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/t=0.8.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/t=0.9.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/t=1.0.png" style="width:100%;">
</div>
