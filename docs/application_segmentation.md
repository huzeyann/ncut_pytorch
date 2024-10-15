# NCUT Tutorial: Segmentation Application

NCUT embedding (eigenvectors) can be discretized to create segmentation mask.


This application output segmentation mask from input 1) all eigenvectors, and input 2) one prompt eigenvector (at a clicked latent pixel). The mask is computed by measuring the cosine similarity between the clicked eigenvector and all the eigenvectors in the latent space, cosine similarity is then normalized, scaled, then threshold to create mask.

0. Compute NCUT eigenvectors.

1. Compute the cosine similarity between the clicked eigenvector and all the eigenvectors in the latent space.

2. Transform the heatmap, normalize and apply scaling (gamma).

3. Threshold the heatmap to get the mask.

4. Optionally de-noise the mask by removing small connected components.

---

### Load Images

<details>
<summary>

Click to expand full code

``` py
from datasets import load_dataset
```

</summary>

``` py linenums="1"
from datasets import load_dataset

dataset = load_dataset('EgoThink/EgoThink', 'Activity', trust_remote_code=True)
dataset = dataset['test']
images = [dataset[i]['image'] for i in range(100)]

import numpy as np
from PIL import Image
import torch

def transform_image(image, resolution=(1024, 1024)):
    image = image.convert('RGB').resize(resolution, Image.LANCZOS)
    # Convert to torch tensor
    image = torch.tensor(np.array(image).transpose(2, 0, 1)).float()
    image = image / 255
    # Normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    return image

images_tsf = torch.stack([transform_image(img) for img in images])
```
</details>

<div style="text-align: center;">
<img src="../images/ego4d_fixing.jpg" alt="ego4d_fixing.jpg" style="width:80%;">
</div>

### Compute NCUT



``` py linenums="1"
from ncut_pytorch import NCUT, rgb_from_tsne_3d
from ncut_pytorch.backbone import load_model, extract_features

model = load_model(model_name="SAM(sam_vit_b)")
model_features = extract_features(images_tsf, model, node_type='attn', layer=11, batch_size=4)
# model_features = model(images_tsf)['attn'][11]  # this also works

num_eig = 30
inp = model_features.reshape(-1, 768) # flatten
eigvectors, eigvalues = NCUT(num_eig=num_eig, device='cuda:0').fit_transform(inp)
tsne_x3d, tsne_rgb = rgb_from_tsne_3d(eigvectors, device='cuda:0')

eigvectors = eigvectors.reshape(-1, 64, 64, num_eig) # (B, H, W, num_eig)
tsne_rgb = tsne_rgb.reshape(-1, 64, 64, 3) # (B, H, W, 3)
```


### Click on a Point

``` py linenums="1"
x1, x2 = 34, 46
clicked_eigvec = eigvectors[0, x1, x2]  # hand pixel
import matplotlib.pyplot as plt
# display the clicked pixel on the tsne_rgb image
plt.imshow(tsne_rgb[0])
plt.scatter(x2, x1, c='red', s=100, label='clicked pixel', edgecolors='black')
plt.legend()
plt.show()
```

<div style="text-align: center;">
<img src="../images/ego4d_click.jpg" alt="ego4d_click.jpg" style="width:60%;">
</div>

### Get Segmentation Mask (correspondence to the clicked point)



``` py linenums="1"
from ncut_pytorch import get_mask
masks = get_mask(eigvectors, clicked_eigvec, threshold=0.5, gamma=1.0, denoise=True, denoise_area_th=3)
```

#### Documentation
::: ncut_pytorch.get_mask
    options:
      heading_level: 3


### Results

``` py linenums="1"
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    axs[0, i].imshow(masks[i])
    axs[0, i].axis('off')
    axs[1, i].imshow(tsne_rgb[i])
    axs[1, i].axis('off')
```

<div style="text-align: center;">
<img src="../images/ego4d_mask.jpg" alt="ego4d_mask.jpg" style="width:100%;">
</div>
