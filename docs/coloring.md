
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

Human perception is not uniform on the RGB color space -- green vs. yellow is less perceptually different than red vs. blue. Therefore, it's a good idea to rotate the RGB cube and try a different color. In the following example, all images has the same euclidean distance matrix, but perceptually they could tell different story.

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