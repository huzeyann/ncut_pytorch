---
hide:
  - toc
---

# Out-of-Distribution Adaptation

NCUT performs well on out-of-distribution (OOD) datasets, demonstrating robust generalization capabilities across diverse image distributions.


#### Training-Free Runtime Adaptation

NCUT adapts to new data at inference time without requiring retraining or fine-tuning—the backbone model remains frozen. (The backbone model is DINO in the examples below.)

The graph affinity matrix automatically adapts to input images, even when they are out-of-distribution. As nodes in the input set contrast against each other, the resulting affinity matrix (analogous to a kernel matrix) dynamically adjusts to accommodate OOD images.

---

##### Example 1
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="../images/gallery_gallery_dataset/dataset1_left.jpg" style="width:60%; object-fit: contain;">
    <img src="../images/gallery_gallery_dataset/dataset1_right.jpg" style="width:40%; object-fit: contain;">
</div>

##### Example 2
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="../images/gallery_gallery_dataset/dataset2_left.jpg" style="width:60%; object-fit: contain;">
    <img src="../images/gallery_gallery_dataset/dataset2_right.jpg" style="width:40%; object-fit: contain;">
</div>

##### Example 3
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="../images/gallery_gallery_dataset/dataset3_left.jpg" style="width:60%; object-fit: contain;">
    <img src="../images/gallery_gallery_dataset/dataset3_right.jpg" style="width:40%; object-fit: contain;">
</div>

##### Example 4
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="../images/gallery_gallery_dataset/dataset4_left.jpg" style="width:60%; object-fit: contain;">
    <img src="../images/gallery_gallery_dataset/dataset4_right.jpg" style="width:40%; object-fit: contain;">
</div>

##### Example 5
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="../images/gallery_gallery_dataset/dataset5_left.jpg" style="width:60%; object-fit: contain;">
    <img src="../images/gallery_gallery_dataset/dataset5_right.jpg" style="width:40%; object-fit: contain;">
</div>