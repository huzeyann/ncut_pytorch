# Make NCUT Aligned across Images


## NCut vs. AlignedCut

<div style="text-align: center;">
<p><b>NCut</b>: Color is <b>not</b> aligned across images. <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">Try on HuggingFace</a>
    <img src="../images/ncut_legacy_vs.jpg" style="width:100%;">
</p>
</div>

**NCut** processes each image independently, with no color alignment -- The arms and legs is colored randomly across different image outputs.


<div style="text-align: center;">
<p><b>AlignedCut</b>: Color is aligned across images. <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">Try on HuggingFace</a>
    <img src="../images/alignedcut_vs.jpg" style="width:100%;">
</p>
</div>

**AlignedCut** improves upon NCut by aligning color across images and handling larger datasets more efficiently. The new developed Nystrom-like approximation solved the scalability and speed bottle-neck.

> [1] AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*,2024
> 
> [2] Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000

---

#### Key Differences:

**NCut**: Independent processing, no color alignment, exact solution.

**AlignedCut**: Aligned processing, color preservation, better scalability.

---

#### Pros (NCut vs. AlignedCut):

- Simple: Uses fewer eigenvectors. The solution space contain less clusters because it's one image.

- Exact: No approximations. Approximation is not necessary for small-scale NCUT.

#### Cons (NCut vs. AlignedCut):

- No Alignment: Color and distance aren't aligned across images. 

- Scalability Issues: Struggles with large pixel counts.








