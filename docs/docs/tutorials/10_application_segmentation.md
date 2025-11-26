# Tutorial 10 - Application - Point Prompting

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
# pip install datasets
from datasets import load_dataset
```

</summary>

``` py linenums="1"
# pip install datasets
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
<img src="/images/tutorials_10_application_segmentation/ego4d_fixing.jpg" alt="ego4d_fixing.jpg" style="width:80%;">
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


## One Point Prompt

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
<img src="/images/tutorials_10_application_segmentation/ego4d_click.jpg" alt="ego4d_click.jpg" style="width:60%;">
</div>

### Get Segmentation Mask (correspondence to the clicked point)



``` py linenums="1"
from ncut_pytorch import get_mask
masks = get_mask(eigvectors, clicked_eigvec, threshold=0.5, gamma=1.0, denoise=True, denoise_area_th=3)
```


### Results: One point

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
<img src="/images/tutorials_10_application_segmentation/ego4d_mask.jpg" alt="ego4d_mask.jpg" style="width:100%;">
</div>

## Multiple Points Prompt

```py linenums="1"

clicks = []  # (i_img, x1, x2)
clicks.append((0, 34, 46))
clicks.append((0, 50, 43))
clicks.append((0, 60, 20))
clicks.append((0, 61, 29))
clicks.append((0, 45, 20))
clicks.append((1, 59, 32))
clicks.append((1, 60, 55))
clicks.append((1, 52, 30))
clicks.append((1, 50, 15))
clicks.append((1, 45, 55))

import matplotlib.pyplot as plt
# display 2 images
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
axs[0].imshow(tsne_rgb[0])
axs[1].imshow(tsne_rgb[1])
for i_img, x1, x2 in clicks:
    axs[i_img].scatter(x2, x1, c='red', s=100, label='clicked pixel', edgecolors='black')

axs[0].legend()
axs[1].legend()
plt.show()
```

<div style="text-align: center;">
<img src="/images/tutorials_10_application_segmentation/ego4d_multi_click.jpg" alt="ego4d_multi_click.jpg" style="width:100%;">
</div>

### How to combine multiple points

Run segmentation separately for every point, then do a | (or) operation to merge the masks.

```py linenums="1"
masks = []
for i_img, x1, x2 in clicks:    
    masks.append(get_mask(eigvectors, eigvectors[i_img, x1, x2], threshold=0.7, gamma=1.0, denoise=True, denoise_area_th=8))
mask = np.stack(masks)
mask = mask.sum(0) > 0
```


### Results: Multiple Point

```py linenums="1"
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    axs[0, i].imshow(mask[i])
    axs[0, i].axis('off')
    axs[1, i].imshow(tsne_rgb[i])
    axs[1, i].axis('off')
```

<div style="text-align: center;">
<img src="/images/tutorials_10_application_segmentation/ego4d_multi_mask.jpg" alt="ego4d_multi_mask.jpg" style="width:100%;">
</div>


## Negative Point Prompt


```py linenums="1"
negative_clicks = []
negative_clicks.append((0, 35, 15))
negative_clicks.append((0, 25, 22))
negative_clicks.append((1, 26, 10))

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
axs[0].imshow(tsne_rgb[0])
axs[1].imshow(tsne_rgb[1])
for i_img, x1, x2 in clicks:
    axs[i_img].scatter(x2, x1, c='red', s=100, label='clicked pixel', edgecolors='black')
for i_img, x1, x2 in negative_clicks:
    axs[i_img].scatter(x2, x1, c='blue', s=100, label='negative clicked pixel', edgecolors='black')
axs[0].legend()
axs[1].legend()
plt.show()
```

<div style="text-align: center;">
<img src="/images/tutorials_10_application_segmentation/ego4d_multi_click_neg.jpg" alt="ego4d_multi_click_neg.jpg" style="width:100%;">
</div>


### How to combine negative points

Run segmentation separately for every point, then do a 'not' operation to merge the positive and negative mask.


``` py linenums="1"
positive_masks, negative_masks = [], []
for i_img, x1, x2 in clicks:    
    mask = get_mask(eigvectors, eigvectors[i_img, x1, x2], threshold=0.7, gamma=1.0, denoise=True, denoise_area_th=10)
    positive_masks.append(mask)
for i_img, x1, x2 in negative_clicks:
    mask = get_mask(eigvectors, eigvectors[i_img, x1, x2], threshold=0.99, gamma=1.0, denoise=False)
    negative_masks.append(mask)
positive_mask = np.stack(positive_masks).sum(0) > 0
negative_mask = np.stack(negative_masks).sum(0) > 0
final_mask = positive_mask & ~negative_mask
```

### Results: Negative Point

``` py linenums="1"
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    axs[0, i].imshow(final_mask[i])
    axs[0, i].axis('off')
    axs[1, i].imshow(tsne_rgb[i])
    axs[1, i].axis('off')
```

<div style="text-align: center;">
<img src="/images/tutorials_10_application_segmentation/ego4d_multi_mask_neg.jpg" alt="ego4d_multi_mask_neg.jpg" style="width:100%;">
</div>

<div style="max-width: 600px; margin: 50px auto; border: 1px solid #ddd; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <a href="https://github.com/huzeyann/ncut_pytorch/tree/master/tutorials" target="_blank" style="text-decoration: none; color: inherit;">
        <div style="display: flex; align-items: center; padding: 15px; background-color: #f6f8fa;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub Logo" style="width: 50px; height: 50px; margin-right: 15px;">
            <div>
                <h2 style="margin: 0;">The complete code for this tutorial</h2>
                <p style="margin: 5px 0 0; color: #586069;">huzeyann/ncut_pytorch</p>
            </div>
        </div>
        <div style="padding: 15px; background-color: #fff;">
            <p style="margin: 0; color: #333;"></p>
        </div>
    </a>
</div>