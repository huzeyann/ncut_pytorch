# How to Get Better Segmentation from NCUT

Please visit our <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">🤗HuggingFace Demo</a>. Play around models and parameters.

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.42.0/gradio.js"
></script>

<gradio-app src="https://huzey-ncut-pytorch.hf.space"></gradio-app>


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

<div style="text-align:left;">
    <pre><code>
eigvecs, eigvals = <span style="color: #003366;">NCUT</span>(num_samples=<span style="color: #A020F0;">30000</span>).fit_transform(inp)
    </code></pre>
</div>


## t-SNE and UMAP parameters

For visualization purpose, t-SNE or UMAP is applied on the NCUT eigenvectors. Our Nystrom approximation is also applied to t-SNE/UMAP. It's possible to in crease the quality of t-SNE/UMAP by tweaking the parameters:

- increase the sample size of Nystrom approximation, `num_samples`

- increase `perplexity` for t-SNE, `n_neighbors` for UMAP

<div style="text-align:left;">
    <pre><code>
<span style="color: #008000;"># balanced speed and quality</span>
X_3d, rgb = <span style="color: #003366;">rgb_from_tsne_3d</span>(eigvecs, num_samples=<span style="color: #A020F0;">300</span>, perplexity=<span style="color: #A020F0;">150</span>)
X_3d, rgb = <span style="color: #003366;">rgb_from_umap_3d</span>(eigvecs, num_samples=<span style="color: #A020F0;">300</span>, n_neighbors=<span style="color: #A020F0;">150</span>, min_dist=<span style="color: #A020F0;">0.1</span>)

<span style="color: #008000;"># extreme quality, much slower</span>
X_3d, rgb = <span style="color: #003366;">rgb_from_tsne_3d</span>(eigvecs, num_samples=<span style="color: #A020F0;">10000</span>, perplexity=<span style="color: #A020F0;">500</span>)
X_3d, rgb = <span style="color: #003366;">rgb_from_umap_3d</span>(eigvecs, num_samples=<span style="color: #A020F0;">10000</span>, n_neighbors=<span style="color: #A020F0;">500</span>, min_dist=<span style="color: #A020F0;">0.1</span>)
    </code></pre>
</div>


Please see [Tutorial - Coloring](coloring.md) for a full comparison of coloring methods:

||Pros|Cons|
|---|---|---|
|t-SNE(3D)|make fuller use of the color space|slow for large samples|
|UMAP(3D)|fast for large samples|holes in the color space; slow for small samples|
|UMAP(sphere)|can be plotted in 2D&3D|do not use the full color space|
|t-SNE(2D)|can be plotted in 2D|do not use the full color space|
|UMAP(2D)|can be plotted in 2D|do not use the full color space|


## Rotate the RGB Cube

Human perception is not uniform on the RGB color space -- green vs. yellow is less perceptually different than red vs. blue. Therefore, it's a good idea to rotate the RGB cube and try a different color. In the following example, all images has the same euclidean distance matrix, but perceptually they could tell different story. Please see [Coloring](coloring.md) for the code to rotate RGB cube.


<div style="text-align: center;">
<img src="../images/color_rotation.png" style="width:100%;">
</div>

## Recursive NCUT

NCUT can be applied recursively, the eigenvectors from previous iteration is the input for the next iteration NCUT. 

<div style="text-align:left;">
    <pre><code>
<span style="color: #008000;"># Type A: cosine, more amplification</span>
eigenvectors1, eigenvalues1 = <span style="color: #003366;">NCUT</span>(num_eig=<span style="color: #A020F0;">100</span>, affinity_focal_gamma=<span style="color: #A020F0;">0.3</span>).fit_transform(input_feats)
eigenvectors2, eigenvalues2 = <span style="color: #003366;">NCUT</span>(num_eig=<span style="color: #A020F0;">50</span>, affinity_focal_gamma=<span style="color: #A020F0;">0.3</span>, distance=<span style="color: #A020F0;">'cosine'</span>, normalize_features=<span style="color: #A020F0;">True</span>).fit_transform(eigenvectors1)
eigenvectors3, eigenvalues3 = <span style="color: #003366;">NCUT</span>(num_eig=<span style="color: #A020F0;">20</span>, affinity_focal_gamma=<span style="color: #A020F0;">0.3</span>, distance=<span style="color: #A020F0;">'cosine'</span>, normalize_features=<span style="color: #A020F0;">True</span>).fit_transform(eigenvectors2)

<span style="color: #008000;"># Type B: euclidean, less amplification, match t-SNE</span>
eigenvectors1, eigenvalues1 = <span style="color: #003366;">NCUT</span>(num_eig=<span style="color: #A020F0;">100</span>, affinity_focal_gamma=<span style="color: #A020F0;">0.3</span>).fit_transform(input_feats)
eigenvectors2, eigenvalues2 = <span style="color: #003366;">NCUT</span>(num_eig=<span style="color: #A020F0;">50</span>, affinity_focal_gamma=<span style="color: #A020F0;">0.3</span>, distance=<span style="color: #A020F0;">'euclidean'</span>, normalize_features=<span style="color: #A020F0;">False</span>).fit_transform(eigenvectors1)
    </code></pre>
</div>



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
