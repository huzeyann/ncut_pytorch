# Make NCUT Aligned across Images

- **NCut** processes each image independently.

- **AlignedCut** process all images as one large-scale graph cut. 

The new developed Nystrom approximation solved the scalability and speed bottle-neck.

## NCut vs. AlignedCut

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


> [1] AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*,2024
> 
> [2] Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000

---

#### Key Differences:

**NCut**: Independent processing, no color alignment, exact solution.

**AlignedCut**: Aligned processing, color preservation, better scalability.

---

#### Pros (NCut):

- Simple: Uses fewer eigenvectors. The solution space contain less clusters because it's one image.

- Exact: No approximations. Approximation is not necessary for small-scale NCUT.

#### Cons (NCut):

- No Alignment: Color and distance aren't aligned across images. 

- Scalability Issues: Struggles with large pixel counts.


#### Limitation of AlignedCut

Adding new images to existing eigenvector solution is not straight-forward, because the eigenvectors for new images need to be consistent with existing images. One solution is use KNN to propagate to new images, please see [Tutorial 3 - Adding Nodes](add_nodes.md).







