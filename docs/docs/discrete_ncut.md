# Discrete NCut

We used the K-Way Ncut algorithm to discretize the clustering results. The discrete Ncut approach means that, instead of relying solely on the continuous partitioning obtained from the standard Ncut formulation, we explicitly assign each pixel feature to a specific category. In other words, it converts the continuous eigenvector representations into discrete cluster labels. As the number of clusters 
𝐾 increases, we can observe noticeable changes in the clustering structure and segmentation boundaries. The discrete Ncut method is particularly valuable in practical applications such as image segmentation and pattern discovery. By transforming continuous clustering information into clear categorical assignments, it enables more interpretable and actionable results. This discretization not only improves the usability of clustering outputs in downstream tasks but also enhances the stability and robustness of the segmentation, especially when dealing with high-dimensional or noisy image data. Overall, discrete Ncut provides a more practical and meaningful way to utilize spectral clustering results in real-world applications.


Example: compute K-way NCut from features
<details>

<summary>
Click to expand full code

``` py
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

</summary>

``` py linenums="1"

import torch
from PIL import Image
import torchvision.transforms as transforms

# DINO v3 model weights URL
DINOV3_URL = "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"

# Load and preprocess image
def preprocess_image(image_path, resolution=(448, 448)):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Extract DINO v3 features
def extract_dinov3_features(image_path, layer=11):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load DINO v3 model
    model = torch.hub.load("facebookresearch/dinov3", "dinov3_vith16plus", 
                           weights=DINOV3_URL)
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)
    
    # Preprocess image
    img_tensor = preprocess_image(image_path).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model.get_intermediate_layers(img_tensor, reshape=True, 
                                                 n=list(range(12)))[layer]
    
    # Convert format: (1, D, H, W) -> (H, W, D)
    features = features.squeeze(0).permute(1, 2, 0).cpu()
    
    print(f"Feature shape: {features.shape}")  # (H, W, D)
    return features

# Usage example
if __name__ == "__main__":
    features = extract_dinov3_features("example.png", layer=11)
    print(f"Extracted feature dimensions: {features.shape}")
    eigvecs = Ncut(n_eig=20).fit_transform(features)  
    # align for discretization-friendly basis
    kway_eigvecs = kway_ncut(eigvecs)
    # cluster assignment and (axis-wise) centroids
    cluster_assignment = kway_eigvecs.argmax(1)
    cluster_centroids = kway_eigvecs.argmax(0) 

```

</details>



The following image is calculated by the features of dinov3_vith16plus. The second line is the discrete NCUT assignments restuls and the third line is the clustering centroid. As we switch between different K, we can see the clustering results become different. The large the K, the more detailed clustering restuls will appear but will also introduce some noise. As you can see the background is divided into different colors, this is because the effect of positional encoding of DINO structure.

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
  <img src="../images/k_8.png" alt="Discrete NCut result for k=8" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
<div class="kway-img k-img-6">
  <img src="../images/k_9.png" alt="Discrete NCut result for k=9" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
<div class="kway-img k-img-7">
  <img src="../images/k_10.png" alt="Discrete NCut result for k=10" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
<div class="kway-img k-img-8">
  <img src="../images/k_11.png" alt="Discrete NCut result for k=11" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
<div class="kway-img k-img-9">
  <img src="../images/k_12.png" alt="Discrete NCut result for k=12" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
</div>
<div class="kway-img k-img-10">
  <img src="../images/k_13.png" alt="Discrete NCut result for k=13" style="width:100%; height:auto; display:block; margin:0 auto; clip-path: inset(15% 0 0 0); -webkit-clip-path: inset(15% 0 0 0);" />
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

From the visual results, it is evident that the choice of 
 k-the number of clusters—plays a crucial role in determining the segmentation granularity. When K is too large, the algorithm over-segments the image, splitting it into many small, fine-grained regions that may correspond to texture variations rather than meaningful semantic parts. Conversely, when K is too small, the segmentation becomes overly coarse, merging distinct areas into broad abstract regions that fail to capture local structure. Therefore, selecting an appropriate K balances detail and interpretability, leading to segmentation maps that align more closely with perceptually coherent regions or objects in the image.

The visualization results below panels labeled “Before K-way” and “After K-way” highlight the difference between the raw eigenvectors produced by the standard NCut algorithm and the axis-aligned eigenvectors obtained after applying the K-way alignment.


(1) Before K-way: The eigenvectors exhibit smooth, continuous variations across the image. The first few eigenvectors are often nearly constant or represent low-frequency global structures, while deeper eigenvectors capture finer spatial variations.

(2) After K-way: Once the K-way alignment is applied, each projection channel becomes more axis-aligned and unimodal, meaning that each cluster now has a dominant direction. This makes the clustering results clearer and easier to discretize. The improved separation between channels directly contributes to more stable and meaningful segmentation outcomes.

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
