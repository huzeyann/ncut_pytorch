
## RGB cube from t-SNE and UMAP 

NCUT eigenvectors are high dimensional, we need a bigger screen to see them all. Or we could use t-SNE/UMAP to reduce the dimension of eigenvectors to 3D, and use a 3D colormap (RGB cube) to show eigenvectors as an RGB image. 

<div class="warning" style='padding:0.1em; background-color:#E9D8FD; color:#69337A'>
<span>
<p style='margin-top:1em; text-align:center'>
<b>PROCEDURE - Coloring</b></p>
<p style='margin-left:1em;'>
1. extract features from DiNOv2 layer9, 20 images, feature shape [20, h, w, 768] </br>
2. compute NCUT eigenvectors, 30 eigenvectors, eigenvector shape [20, h, w, 30] </br>
3. use t-SNE or UMAP to reduce 30 eigenvectors to 3D, shape [20, h, w, 3] </br>
4. color each pixel by 3D colormap (RGB cube)
</p>
</p></span>
</div>

<div style="text-align: center;">
<img src="../images/spectral_tsne_3d.png" style="width:100%;">
</div>
<div style="text-align: center;">
<img src="../images/spectral_tsne_3d_images.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/spectral_umap_sphere.png" style="width:100%;">
</div>
<div style="text-align: center;">
<img src="../images/spectral_umap_sphere_images.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/spectral_umap_3d.png" style="width:100%;">
</div>
<div style="text-align: center;">
<img src="../images/spectral_umap_3d_images.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/spectral_tsne_2d.png" style="width:40%;">
</div>
<div style="text-align: center;">
<img src="../images/spectral_tsne_2d_images.png" style="width:100%;">
</div>

<div style="text-align: center;">
<img src="../images/spectral_umap_2d.png" style="width:40%;">
</div>
<div style="text-align: center;">
<img src="../images/spectral_umap_2d_images.png" style="width:100%;">
</div>

## RGB cube rotation

Human perception is not uniform on the RGB color space -- green vs. yellow is less perceptionally different than red vs. blue. Therefore, it's a good idea to rotate the RGB cube and try a different color.

```py linenums="1"
# 9-way rotation of the rgb
import matplotlib.pyplot as plt
from ncut_pytorch import rotate_rgb_cube

fig, axs = plt.subplots(2, 3, figsize=(6, 4))
axs = axs.flatten()
for i in range(6):
    _rgb = rotate_rgb_cube(rgb[4], position=i+1)
    ax[i].imshow(_rgb)
    ax[i].axis("off")
plt.suptitle("Rotation of the RGB cube")
plt.tight_layout()
plt.show()
```

<div style="text-align: center;">
<img src="../images/color_rotation.png" style="width:100%;">
</div>
