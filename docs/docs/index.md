

<div style="text-align: center;">
  <img src="/images/index/Ncut.svg" alt="Ncut" style="width: 80%; filter: brightness(60%) grayscale(100%);"/>
</div>


<div style="display: flex; justify-content: center; margin-top: 20px; flex-wrap: wrap;">

  <a href="https://github.com/huzeyann/ncut_pytorch" target="_blank" style="width: 30%; text-align: center; background-color: #007BFF; color: white; padding: 10px; border-radius: 5px; margin-right: 2%; margin-bottom: 10px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="width: 24px; height: 24px; vertical-align: middle;"/> GitHub
  </a>

  <a href="https://huggingface.co/spaces/huzey/Ncut-pytorch" target="_blank" style="width: 30%; text-align: center; background-color: #FF5733; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-right: 2%; margin-bottom: 10px;">
    🤗 Demo
  </a>

  <div style="width: 30%; text-align: center; background-color: #6C757D; color: white; padding: 10px 20px; border-radius: 5px; margin-bottom: 10px;">
    📝 Paper (Coming)
  </div>

</div>

<style>
  @media (max-width: 768px) {
    div[style*="display: flex"] a,
    div[style*="display: flex"] div {
      width: 100%;
      margin-right: 0;
    }
  }
</style>

<div style="width: 60%; text-align: center; margin:0 auto;">
    <pre><code class="language-shell">pip install Ncut-pytorch</code></pre>
</div>




# Ncut: Nyström Normalized Cut

**Normalized Cut**, a.k.a. spectral clustering, is a graphical method to analyze data grouping in the affinity eigenvector space. It has been widely used for unsupervised segmentation in the 2000s.

**Nyström Normalized Cut**, is a new approximation algorithm developed for large-scale graph cuts, $O(n)$ time complexity, $O(1)$ space complexity. Solve million-scale graph in milliseconds.

<div  style="text-align: center;">
<video width="90%" controls muted autoplay loop>
  <source src="/images/index/Ncut_video_sam_264_small.mp4" type="video/mp4">
</video>
<p>Video: Ncut applied to image encoder features from Segment Anything Model. </br> RGB color is 3D spectral-tSNE embedding of Ncut eigenvectors.
<a href="/gallery/gallery_sam_video/">code</a> **TODO: update this code**
</p>
</div>

<!-- <div style="text-align: center;">
<img src="https://github.com/user-attachments/assets/a5d8a966-990b-4f6d-be10-abb00291bee2" style="width:100%;">
<p>Ncut (DINO features as input) produce segmentation at various granularity.
</p>
</div> -->




<div  style="text-align: center;">
<video width="90%" controls muted autoplay loop>
  <source src="/images/demo_heatmap.mp4" type="video/mp4">
</video>
<p>Video: Heatmap is cosine distance of eigenvectors, w.r.t the mouse pointer.
<a href="/alignedcut_vs_Ncut/">details</a> TODO: change video to the new hierarchy segmentation method
</p>
</div>

<!-- ## Demo -->

<!-- Please visit the <a href="https://huggingface.co/spaces/huzey/Ncut-pytorch" target="_blank">🤗HuggingFace Demo</a>. Play around datasets and models. -->


<!--
<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.42.0/gradio.js"
></script>
-->

<gradio-app src="https://huzey-Ncut-pytorch.hf.space"></gradio-app>

<script>
	function handleLoadComplete() {
		console.log("Embedded space has finished rendering");
		// Click the "Gallery" button
		const galleryButton = document.querySelector("button:contains('Gallery')");
		if (galleryButton) {
			galleryButton.click();
		} else {
			console.log("Gallery button not found");
		}
	}

	const gradioApp = document.querySelector("gradio-app");
	if (gradioApp) {
		gradioApp.addEventListener("render", handleLoadComplete);
	} else {
		console.log("gradio-app element not found");
	}
</script>

<!-- <iframe
	src="https://huzey-Ncut-pytorch.hf.space"
	frameborder="0"
	width="100%"
	height="800"
></iframe> -->

## Gallery
Just plugin features extracted from any pre-trained model and ready to go. Ncut works for any input -- image, text, video, 3D, .... Planty examples code and plots in the [Gallery](/gallery/)

<div style="text-align: center;">
<a href="/gallery/">
<img src="/images/index/Ncut_gallery_cover.jpg" style="width:100%;">
</a>
</div>

---

## Installation

<div style="text-align:">
    <pre><code class="language-shell">pip install -U ncut-pytorch</code></pre>
</div>


---

## Quick Start: plain Ncut


```py linenums="1"
import torch
from ncut_pytorch import Ncut

features = torch.rand(1960, 768)
eigvecs = Ncut(n_eig=20).fit_transform(features)
  # (1960, 20)

from ncut_pytorch import kway_ncut
kway_eigvecs = kway_ncut(eigvecs)
cluster_assignment = kway_eigvecs.argmax(1)
cluster_centroids = kway_eigvecs.argmax(0)
```

## Quick Start: Ncut DINOv3 Predictor

```py linenums="1"
from ncut_pytorch.predictor import NcutDinov3Predictor
from PIL import Image

predictor = NcutDinov3Predictor(model_cfg="dinov3_vitl16")
predictor = predictor.to('cuda')

images = [Image.open(f"images/view_{i}.jpg") for i in range(4)]
predictor.set_images(images)

image = predictor.summary(n_segments=[10, 25, 50, 100], draw_border=True)
display(image)

```

![summary](https://github.com/user-attachments/assets/a5d8a966-990b-4f6d-be10-abb00291bee2)

---

> paper in prep, Yang 2025
>
> AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*,2024
> 
> Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
