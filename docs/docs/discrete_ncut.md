# Discrete NCut

We have used K-Way Ncut to discretize the clustering results. We can see the clustering results are changing while increasing K.

- kway_ncut uses axis_align to orthogonally align the continuous eigenvectors so that each class dominates one axis (i.e., `eigvec @ R`).
- axis_align performs subsampling (farthest_point_sampling), L2 normalization, initializes R from k farthest points, then iterates: project → argmax one-hot discretize → SVD (`D^T F`) → update `R = V U^T`, minimizing the NCut objective \(2\,(n - \sum S)\) until convergence.
- Final labels are obtained by argmax over rows (the index of the one-hot maximum).

The following image is calculated by the features of DINO V2.

<div class="kway-tabs" style="text-align:center;">
  <input type="radio" id="k5" name="k" checked>
  <label for="k5" class="kbtn">k=5</label>
  <input type="radio" id="k6" name="k">
  <label for="k6" class="kbtn">k=6</label>
  <input type="radio" id="k7" name="k">
  <label for="k7" class="kbtn">k=7</label>
  <input type="radio" id="k8" name="k">
  <label for="k8" class="kbtn">k=8</label>
  <input type="radio" id="k9" name="k">
  <label for="k9" class="kbtn">k=9</label>
  <input type="radio" id="k10" name="k">
  <label for="k10" class="kbtn">k=10</label>

<div class="kway-img k-img-5">
  <img src="../images/k_5.png" alt="Discrete NCut result for k=5" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
<div class="kway-img k-img-6">
  <img src="../images/k_6.png" alt="Discrete NCut result for k=6" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
<div class="kway-img k-img-7">
  <img src="../images/k_7.png" alt="Discrete NCut result for k=7" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
<div class="kway-img k-img-8">
  <img src="../images/k_8.png" alt="Discrete NCut result for k=8" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
<div class="kway-img k-img-9">
  <img src="../images/k_9.png" alt="Discrete NCut result for k=9" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
<div class="kway-img k-img-10">
  <img src="../images/k_10.png" alt="Discrete NCut result for k=10" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
</div>
<style>
.kway-tabs input[type="radio"]{display:none;}
/* Default: hide all, show the selected image when radio works */
.kway-tabs .kway-img{display:none;}
#k5:checked ~ .k-img-5{display:block;}
#k6:checked ~ .k-img-6{display:block;}
#k7:checked ~ .k-img-7{display:block;}
#k8:checked ~ .k-img-8{display:block;}
#k9:checked ~ .k-img-9{display:block;}
#k10:checked ~ .k-img-10{display:block;}
.kbtn{display:inline-block; padding:6px 12px; border:1px solid var(--md-default-fg-color--lighter, #ccc); border-radius:6px; margin:0 4px; cursor:pointer;}
#k5:checked + label.kbtn, #k6:checked + label.kbtn, #k7:checked + label.kbtn, #k8:checked + label.kbtn, #k9:checked + label.kbtn, #k10:checked + label.kbtn{background: var(--md-primary-fg-color, #3f51b5); color: #fff; border-color: transparent;}
</style>
<style>
/* Enhance toggle buttons look */
.kway-toggle-bar{display:inline-flex; align-items:center; gap:6px;}
.kway-toggle-bar .md-button{border:1px solid var(--md-default-fg-color--lighter, #ccc); border-radius:6px; background: var(--md-default-bg-color, transparent); color: var(--md-default-fg-color, inherit); cursor:pointer; user-select:none; min-width: 140px;} 
.kway-toggle-bar .md-button--primary{background: var(--md-primary-fg-color, #3f51b5); color:#fff; border-color: transparent;}
.kway-toggle-bar .md-button:hover{filter: brightness(0.95);} 
.kway-toggle-bar .md-button:active{transform: translateY(1px);} 
</style>

We can see from the results that K should be an appropriate number. Large K tends to segment the images into more blocks while small K will only show an abstract segmentation restul of the feature space.


Example: compute K-way NCut from features
```python
import torch
from ncut_pytorch import Ncut, kway_ncut

# features: shape (n, d)
features = torch.rand(1960, 768)

# continuous eigenvectors from NCut, shape (n, k)
eigvecs = Ncut(n_eig=20).fit_transform(features)  # (1960, 20)

# align for discretization-friendly basis
kway_eigvecs = kway_ncut(eigvecs)

# cluster assignment and (axis-wise) centroids
cluster_assignment = kway_eigvecs.argmax(1)
cluster_centroids = kway_eigvecs.argmax(0) 
```


<div id="kway-toggle" style="text-align:center;">
  <input type="radio" id="view-before" name="kview" checked>
  <label for="view-before" class="md-button kview-btn">Before k-way</label>
  <input type="radio" id="view-after" name="kview">
  <label for="view-after" class="md-button kview-btn">After k-way</label>

<div id="kway-before" class="kview-panel">
<p><strong>Before k-way (NCut eigenvectors)</strong></p>
<p>The first row is theoretically near-constant; deeper rows have higher spatial frequency.</p>
<div style="text-align:center;">
<img src="../images/ncut_batch_eigenvectors.png" alt="NCut eigenvectors (before k-way)" style="max-width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(10% 0 0 0);" />
</div>
</div>

<div id="kway-after" class="kview-panel">
<p><strong>After k-way (K-way projection channels, k=10)</strong></p>
<p>These are the 10 channel responses before one-hot; after alignment, channels become more axis-aligned (unimodal).</p>
<div style="text-align:center;">
<img src="../images/ncut_kway_all_dimensions.png" alt="K-way eigenvectors channels (k=10), before argmax" style="max-width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(10% 0 0 0); -webkit-clip-path: inset(10% 0 0 0);" />
</div>
</div>
</div>
<style>
#kway-toggle input[type="radio"]{display:none;}
#kway-toggle .kview-btn{display:inline-block; padding:6px 14px; margin:0 4px 8px 4px; border:1px solid var(--md-default-fg-color--lighter, #ccc); border-radius:6px; cursor:pointer; user-select:none; min-width:140px;}
#view-before:checked + label.kview-btn{background: var(--md-primary-fg-color, #3f51b5); color:#fff; border-color: transparent;}
#view-after:checked + label.kview-btn{background: var(--md-primary-fg-color, #3f51b5); color:#fff; border-color: transparent;}
.kview-panel{display:none;}
#view-before:checked ~ #kway-before{display:block;}
#view-after:checked ~ #kway-after{display:block;}
</style>
