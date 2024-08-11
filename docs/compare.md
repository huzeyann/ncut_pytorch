### Comparison


#### NCUT+t-SNE vs. PCA+t-SNE

<div class="warning" style='padding:0.1em; background-color:#E9D8FD; color:#69337A'>
<span>
<p style='margin-top:1em; text-align:center'>
<b>PROCEDURE - Comparison</b></p>
<p style='margin-left:1em;'>
<b>For NCUT+t-SNE:</b> </br>
1. extract features from DiNOv2 layer9, 20 images, feature shape [20, 32, 32, 768] </br>
2. compute NCUT eigenvectors, 20 eigenvectors, eigenvector shape [20, 32, 32, 20] </br>
3. use t-SNE or UMAP to reduce 20 eigenvectors to 3D, shape [20, 32, 32, 3] </br>
4. color each pixel by 3D colormap (RGB cube) </br>
<b>For PCA+t-SNE:</b> </br>
1. extract features from DiNOv2 layer9, 20 images, feature shape [20, 32, 32, 768] </br>
2. compute PCA embeddings, 20 PCs, PC shape [20, 32, 32, 20] </br>
3. use t-SNE or UMAP to reduce 20 PCs to 3D, shape [20, 32, 32, 3] </br>
4. color each pixel by 3D colormap (RGB cube) </br>
</p>
</p></span>
</div>

<div style="text-align: center;">
<img src="/images/pca/tsne_n_eig=20.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="/images/pca/tsne_n_pca=20.png" style="width:100%;">
</div>
<hr>

<div style="text-align: center;">
<img src="/images/pca/tsne_n_eig=50.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="/images/pca/tsne_n_pca=50.png" style="width:100%;">
</div>
<hr>

#### NCUT+UMAP vs. PCA+UMAP

<div style="text-align: center;">
<img src="/images/pca/umap_n_eig=20.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="/images/pca/umap_n_pca=20.png" style="width:100%;">
</div>
<hr>

<div style="text-align: center;">
<img src="/images/pca/umap_n_eig=50.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="/images/pca/umap_n_pca=50.png" style="width:100%;">
</div>
<hr>

#### raw NCUT vs. raw PCA


<div style="text-align: center;">
<img src="/images/raw_ncut_24.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="/images/raw_pca_24.png" style="width:100%;">
</div>
