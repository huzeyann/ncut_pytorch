---
hide:
  - toc
---

# Application Gallery

This application output segmentation mask given prompt points, it work by computing and thresholding L2 distance on the NCUT embedding:

1. Compute NCUT embedding, spectral-tSNE. Color is aligned across all images [Aligned NCUT](alignedcut_vs_ncut.md)

2. Take rgb value (NCUT embedding) at prompt pixel, compute L2 distance to all other pixels

3. Threshold L2 distance to get segmentation mask

4. De-noise the mask, combine multiple prompt points.

---

You can try this demo segmentation app at <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">🤗HuggingFace</a> (Switch to the Tab "Application").

##### Example 1

<div  style="text-align: center;">
<video width="100%" controls muted autoplay loop>
  <source src="../images/app_demo_0.mp4" type="video/mp4">
</video>
</div>

##### Example 2

<div  style="text-align: center;">
<video width="100%" controls muted loop>
  <source src="../images/app_demo_1.mp4" type="video/mp4">
</video>
</div>

##### Example 3

<div  style="text-align: center;">
<video width="100%" controls muted loop>
  <source src="../images/app_demo_2.mp4" type="video/mp4">
</video>
</div>

##### Example 4

<div  style="text-align: center;">
<video width="100%" controls muted loop>
  <source src="../images/app_demo_3.mp4" type="video/mp4">
</video>
</div>