# Make NCUT Aligned across Images

- **NCut** processes each image independently.

- **AlignedCut** process all images as one large-scale graph cut. 

AlignedCut means concatenating images and make a large-scale graph. The new developed Nystrom approximation solved the scalability and speed bottle-neck.

## NCut vs. AlignedCut

In this example, for NCut, the color of that 'Gibson Les Pual' guitar is arbitrary across images. For AlignedCut, the guitar is colored consistent across images.

<div style="text-align: center;">
<p><b>NCut</b>: Color is <b>not</b> aligned across images. <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">Try on HuggingFace</a>
    <img src="../images/ncut_legacy_vs.jpg" style="width:100%;">
</p>
</div>


<div style="text-align: center;">
<p><b>AlignedCut</b>: Color is aligned across images. <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">Try on HuggingFace</a>
    <img src="../images/alignedcut_vs.jpg" style="width:100%;">
</p>
</div>

---

## Correspondence from AlignedCut

Since the color is consistent across images, we could build a simple software: it checks the distance (in eigenvector color) from one selected pixel to all the other pixels.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gllutIdACcEHtJ81n_tGVNgR6fTupV46): A demo software for one-point prompting segmentation and pseudo-labeling.

<div  style="text-align: center;">
<video width="90%" controls muted autoplay loop>
  <source src="../images/demo_heatmap.mp4" type="video/mp4">
</video>
<p>Video: Heatmap is cosine distance of eigenvectors, w.r.t the mouse pointer.
<a href="https://colab.research.google.com/drive/1gllutIdACcEHtJ81n_tGVNgR6fTupV46" target="_blank">code</a>
</p>
</div>




---

## Pros and Cons

#### Pros (NCut):

- Simple: Uses fewer eigenvectors. The solution space contain less clusters because it's one image.

- Exact: No approximations. Approximation is not necessary for small-scale NCUT.

#### Cons (NCut):

- No Alignment: Color and distance aren't aligned across images. 

- Scalability Issues: Struggles with large pixel counts.


## Limitation of AlignedCut

Adding new images to existing eigenvector solution is not straight-forward, because the eigenvectors for new images need to be consistent with existing images. One solution is use KNN to propagate to new images, please see [Tutorial 3 - Adding Nodes](add_nodes.md).








> [1] AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*,2024
> 
> [2] Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
