# How to Get Better Segmentation from NCUT

## Pick Number of Eigenvectors

More eigenvectors give more details on the segmentation, less eigenvectors is more robust.

In NCUT, by math design (see [How NCUT Works](how_ncut_works.md)), i-th eigenvector give a optimal graph cut that divides the graph into 2^(i-1) clusters. E.g. `eigenvectors[:, 1]` divides the graph into 2 clusters, `eigenvectors[:, 1:4]` divides the graph into 8 clusters.

To answer the question ``How many eigenvectors do I need?'', one need to consider the complexity of the graph (how many images, what's the backbone model), and the goal (e.g., whole-body or body parts). 

A general rule-of-thumb is:

1. Less eigenvectors for robust results, more eigenvectors can be noisy

2. More eigenvectors for larger set of images, less eigenvectors for fewer images

3. More eigenvectors for objects parts, less eigenvectors for whole-object

4. Try [recursive NCUT](how_to_get_better_segmentation.md/#recursive-ncut-for-small-object-parts) for large set of images

Here's an example grid search of how many eigenvectors to include:

<div style="text-align: center;">
<img src="../images/n_eig_raw.jpg" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/n_eig_tsne.jpg" style="width:100%;">
</div>



## Clean up the Affinity Matrix 

When an graph has noisy connections, e.g. random weak connections across any pair of nodes, one could apply thresholding to get rid of this type of noise.

The `0 < affinity_focal_gamma <= 1` parameter is controlling the smooth thresholding, each element `A_ij` in affinity is transformed as:

```
A_ij = 1 - cos_similarity(node_i, node_j)
A_ij = exp(-(A_ij / affinity_focal_gamma))
```

In practice, it's recommended to reduce the value of `affinity_focal_gamma` (default 0.3) until the segmentation is shredded too much. 

<div style="text-align: center;">
<img src="../images/parameters/t=0.1.png" style="width:100%;">
</div>


<div style="text-align: center;">
<img src="../images/parameters/t=0.3.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/parameters/t=0.9.png" style="width:100%;">
</div>

## Increase Sampling Size

It's recommended to use as large sample size as it fits into memory (default `num_sample=3000`).

Nystrom approximation made it possible to compute on large-scale graph (see [How NCUT Works](how_ncut_works.md)). A decent sampling size of Nystrom approximation is critical for a good approximation. 
In general, as the graph gets larger, the sampling size need to be increased. In fact, the increased need for larger sampling size is due to the increased complexity of the graph but not more number of nodes.
E.g., 100 images of different cats will be more complex than 100 views of the same cats, hence larger sample size is recommended.
Thank's to [svd_lowrank](https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html), the time complexity of NCUT scales **linearly** w.r.t. sample size, the bottleneck is memory scales quadratically. 

On a 16GB GPU, `num_sample=30000` fits into memory. On a 12GB GPU, `num_sample=20000` fits into memory.

## Rotate the RGB Cube

Human perception is not uniform on the RGB color space -- green vs. yellow is less perceptually different than red vs. blue. Therefore, it's a good idea to rotate the RGB cube and try a different color. In the following example, all images has the same euclidean distance matrix, but perceptually they could tell different story. Please see [Coloring](coloring.md) for the code to rotate RGB cube.


<div style="text-align: center;">
<img src="../images/color_rotation.png" style="width:100%;">
</div>

## Recursive NCUT for Small Object Parts

NCUT can be applied recursively, the eigenvectors from previous iteration is the input for the next iteration NCUT. 

```py linenums="1"
# Type A: cosine, more amplification
eigenvectors1, eigenvalues1 = NCUT(num_eig=100, affinity_focal_gamma=0.3).fit_transform(input_feats)
eigenvectors2, eigenvalues2 = NCUT(num_eig=50, affinity_focal_gamma=0.3, distance='cosine', normalize_features=True).fit_transform(eigenvectors1)
eigenvectors3, eigenvalues3 = NCUT(num_eig=20, affinity_focal_gamma=0.3, distance='cosine', normalize_features=True).fit_transform(eigenvectors2)

# Type B: euclidean, less amplification, match t-SNE
eigenvectors1, eigenvalues1 = NCUT(num_eig=100, affinity_focal_gamma=0.3).fit_transform(input_feats)
eigenvectors2, eigenvalues2 = NCUT(num_eig=50, affinity_focal_gamma=0.3, distance='euclidean', normalize_features=False).fit_transform(eigenvectors1)
```

**Recursive NCUT amplifies small object parts** because:

1. Eigenvectors with small eigenvalues are amplified -- we didn't take eigenvalue but only took eigenvectors in each iteration. This could be beneficial if we picked a good eigenvector number, if too much eigenvectors are picked, the noise will gets amplified. Less or no recursion is better for whole-object segmentation.
2. Affinity is amplified in each iteration. When used recursive NCUT with `affinity_focal_gamma < 1`, each iteration will clean up the affinity graph and amplify the strong connections.

The following example applied NCUT to 300 images, recursion 1 take 100 eigenvectors, recursion 2 take 50 eigenvectors, recursion 3 take 20 eigenvectors. 

<div style="text-align: center;">
<img src="../images/recursion_L1.jpg" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/recursion_L2.jpg" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/recursion_L3.jpg" style="width:100%;">
</div>
