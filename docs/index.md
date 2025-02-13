

<div style="text-align: center;">
  <img src="./images/ncut.svg" alt="NCUT" style="width: 80%; filter: brightness(60%) grayscale(100%);"/>
</div>


<div style="display: flex; justify-content: center; margin-top: 20px; flex-wrap: wrap;">

  <a href="https://github.com/huzeyann/ncut_pytorch" target="_blank" style="width: 30%; text-align: center; background-color: #007BFF; color: white; padding: 10px; border-radius: 5px; margin-right: 2%; margin-bottom: 10px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="width: 24px; height: 24px; vertical-align: middle;"/> GitHub
  </a>

  <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank" style="width: 30%; text-align: center; background-color: #FF5733; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-right: 2%; margin-bottom: 10px;">
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
    <pre><code class="language-shell">pip install ncut-pytorch</code></pre>
</div>




# NCUT: Nyström Normalized Cut

**Normalized Cut**, aka. spectral clustering, is a graphical method to analyze data grouping in the affinity eigenvector space. It has been widely used for unsupervised segmentation in the 2000s.

**Nyström Normalized Cut**, is a new approximation algorithm developed for large-scale graph cuts,  a large-graph of million nodes can be processed in under 10s (cpu) or 2s (gpu).  

<div  style="text-align: center;">
<video width="90%" controls muted autoplay loop>
  <source src="./images/ncut_video_sam_264_small.mp4" type="video/mp4">
</video>
<p>Video: NCUT applied to image encoder features from Segment Anything Model. </br> RGB color is 3D spectral-tSNE embedding of NCUT eigenvectors.
<a href="./gallery_sam_video">code</a>
</p>
</div>

<div style="text-align: center;">
<img src="./images/ncut_hierarchy_vs_sam.jpg" style="width:100%;">
<p>NCUT on DiNO features segmentation at various granularity. </br> NCUT segments coloring is aligned across images, SAM color is arbitrary.
</p>
</div>

<div class="warning" style='padding:0.1em; background-color:#E9D8FD; color:#69337A'>
<span>
<p style='margin-top:1em; text-align:center'>
<b>PROCEDURE <a href="./how_ncut_works">How NCUT Works</a></b></p>
<p style='margin-left:1em;'>
1. <b>Feature Extraction</b>: extract feature from pre-trained model.</br>
2. <b>NCUT</b>: compute 100 NCUT eigenvectors, input feature is from deep models. </br>
</p>
</p></span>
</div>



**Demo Application**: Point-Prompting Segmentation tool for pseudo-labeling across multiple images. Try it in <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">🤗HuggingFace</a> (Switch to the Tab "Application"). More Examples in [Gallery](gallery_application.md).


<div  style="text-align: center;">
<video width="90%" controls muted autoplay loop>
  <source src="./images/demo_heatmap.mp4" type="video/mp4">
</video>
<p>Video: Heatmap is cosine distance of eigenvectors, w.r.t the mouse pointer.
<a href="./alignedcut_vs_ncut">details</a>
</p>
</div>

<!-- ## Demo -->

<!-- Please visit the <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">🤗HuggingFace Demo</a>. Play around datasets and models. -->


<!--
<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.42.0/gradio.js"
></script>
-->

<gradio-app src="https://huzey-ncut-pytorch.hf.space"></gradio-app>

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
	src="https://huzey-ncut-pytorch.hf.space"
	frameborder="0"
	width="100%"
	height="800"
></iframe> -->

## Gallery
Just plugin features extracted from any pre-trained model and ready to go. NCUT works for any input -- image, text, video, 3D, .... Planty examples code and plots in the [Gallery](gallery.md)

<div style="text-align: center;">
<a href="./gallery/">
<img src="./images/ncut_gallery_cover.jpg" style="width:100%;">
</a>
</div>

---

## Installation

#### 1. Install PyTorch

<div style="text-align:">
<pre><code class="language-shell">conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
</code></pre>
</div>

#### 2. Install `ncut-pytorch`

<div style="text-align:">
    <pre><code class="language-shell">pip install ncut-pytorch</code></pre>
</div>


#### Installation Trouble Shooting

In case of `pip` install <a style="color: red;">error</a>, please try install the build dependencies.

Option A:
<div style="text-align:">
    <pre style="display: inline;"><code class="language-shell">sudo apt-get update && sudo apt-get install build-essential cargo rustc -y</code></pre>
</div>

Option B:
<div style="text-align:">
    <pre><code class="language-shell">conda install rust -c conda-forge</code></pre>
</div>

Option C:
<div style="text-align:">
    <pre><code class="language-shell">curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && . "$HOME/.cargo/env"</code></pre>
</div>


---

## Quick Start

Minimal example on how to run NCUT, more examples in [Tutorial](tutorials.md) and [Gallery](gallery.md).

<div style="text-align:left;">
    <pre><code>
<span style="color: #008000;"><b>import</b></span> torch
<span style="color: #008000;"><b>from</b></span> ncut_pytorch <span style="color: #008000;"><b>import</b></span> <span style="color: #FF6D00;">NCUT</span>, rgb_from_tsne_3d

model_features = torch.rand(20, 64, 64, 768)  <span style="color: #008080;"># (B, H, W, C)</span>

inp = model_features.<span style="color: #008080;">reshape</span>(-1, 768)  <span style="color: #008080;"># flatten</span>
eigvectors, eigvalues = <span style="color: #FF6D00;">NCUT</span>(num_eig=100, device=<span style="color: #ab38f2;">'cuda:0'</span>).fit_transform(inp)
tsne_x3d, tsne_rgb = rgb_from_tsne_3d(eigvectors, device=<span style="color: #ab38f2;">'cuda:0'</span>)

eigvectors = eigvectors.<span style="color: #008080;">reshape</span>(20, 64, 64, 100)  <span style="color: #008080;"># (B, H, W, num_eig)</span>
tsne_rgb = tsne_rgb.<span style="color: #008080;">reshape</span>(20, 64, 64, 3)  <span style="color: #008080;"># (B, H, W, 3)</span>
    </code></pre>
</div>

#### Load Feature Extractor Model

Any backbone model works as plug-in feature extractor. 
We have implemented some backbone models, here is a list of available models:

<div style="text-align:left;"> <pre><code> 
<span style="color: #008000;"><b>from</b></span> ncut_pytorch.backbone <span style="color: #008000;"><b>import</b></span> list_models 
<span style="color: #008080;">print</span>(list_models()) 
<span style="color: #808080;">[
  'SAM2(sam2_hiera_t)', 'SAM2(sam2_hiera_s)', 'SAM2(sam2_hiera_b+)', 'SAM2(sam2_hiera_l)', 
  'SAM(sam_vit_b)', 'SAM(sam_vit_l)', 'SAM(sam_vit_h)', 'MobileSAM(TinyViT)', 
  'DiNOv2reg(dinov2_vits14_reg)', 'DiNOv2reg(dinov2_vitb14_reg)', 'DiNOv2reg(dinov2_vitl14_reg)', 'DiNOv2reg(dinov2_vitg14_reg)', 
  'DiNOv2(dinov2_vits14)', 'DiNOv2(dinov2_vitb14)', 'DiNOv2(dinov2_vitl14)', 'DiNOv2(dinov2_vitg14)', 
  'DiNO(dino_vits8_896)', 'DiNO(dino_vitb8_896)', 'DiNO(dino_vits8_672)', 'DiNO(dino_vitb8_672)', 'DiNO(dino_vits8_448)', 'DiNO(dino_vitb8_448)', 'DiNO(dino_vits16_448)', 'DiNO(dino_vitb16_448)',
  'Diffusion(stabilityai/stable-diffusion-2)', 'Diffusion(CompVis/stable-diffusion-v1-4)', 'Diffusion(stabilityai/stable-diffusion-3-medium-diffusers)',
  'CLIP(ViT-B-16/openai)', 'CLIP(ViT-L-14/openai)', 'CLIP(ViT-H-14/openai)', 'CLIP(ViT-B-16/laion2b_s34b_b88k)', 
  'CLIP(convnext_base_w_320/laion_aesthetic_s13b_b82k)', 'CLIP(convnext_large_d_320/laion2b_s29b_b131k_ft_soup)', 'CLIP(convnext_xxlarge/laion2b_s34b_b82k_augreg_soup)', 
  'CLIP(eva02_base_patch14_448/mim_in22k_ft_in1k)', "CLIP(eva02_large_patch14_448/mim_m38m_ft_in22k_in1k)",
  'MAE(vit_base)', 'MAE(vit_large)', 'MAE(vit_huge)', 
  'ImageNet(vit_base)'
]</span>
<span style="color: #008000;"><b>from</b></span> ncut_pytorch.backbone_text <span style="color: #008000;"><b>import</b></span> list_models 
<span style="color: #008080;">print</span>(list_models()) 
<span style="color: #808080;">["meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3-8B", "gpt2"]</span>
</span> </code></pre> </div>

#### Image model example:

<div style="text-align:left;"> <pre><code> <span style="color: #008000;">
<b>import</b></span> torch <span style="color: #008000;"><b>from</b></span> ncut_pytorch <span style="color: #008000;"><b>import</b></span> <span style="color: #FF6D00;">NCUT</span>, rgb_from_tsne_3d 
<span style="color: #008000;"><b>from</b></span> ncut_pytorch.backbone <span style="color: #008000;"><b>import</b></span> load_model, extract_features

model = load_model(model_name=<span style="color: #ab38f2;">"SAM(sam_vit_b)"</span>) 
images = torch.rand(20, 3, 1024, 1024) 
model_features = extract_features(images, model, node_type=<span style="color: #ab38f2;">'attn'</span>, layer=<span style="color: #008080;">6</span>) <span style="color: #008080;">
# model_features = model(images)['attn'][6]  # this also works</span>

inp = model_features.<span style="color: #008080;">reshape</span>(-1, 768) <span style="color: #008080;"># flatten</span>
eigvectors, eigvalues = <span style="color: #FF6D00;">NCUT</span>(num_eig=100, device=<span style="color: #ab38f2;">'cuda:0'</span>).fit_transform(inp) 
tsne_x3d, tsne_rgb = rgb_from_tsne_3d(eigvectors, device=<span style="color: #ab38f2;">'cuda:0'</span>)

eigvectors = eigvectors.<span style="color: #008080;">reshape</span>(20, 64, 64, 100) <span style="color: #008080;"># (B, H, W, num_eig)</span> 
tsne_rgb = tsne_rgb.<span style="color: #008080;">reshape</span>(20, 64, 64, 3) <span style="color: #008080;"># (B, H, W, 3)</span> </code></pre>
</div>

#### Text model example:


<details>
<summary>

This example use your access token and download Llama from HuggingFace. How to set up Llama access token from HuggingFace (click to expand):

</summary>

<p>Step 1: Request access for Llama from <a ref="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B" target="_blank">https://huggingface.co/meta-llama/Meta-Llama-3.1-8B</a>

<p>Step 2: Find your access token at <a ref="https://huggingface.co/settings/tokens" target="_blank">https://huggingface.co/settings/tokens</a> </p>

</details>

<div style="text-align:left;">
    <pre><code>
<span style="color: #008000;"><b>import</b></span> os
<span style="color: #008000;"><b>from</b></span> ncut_pytorch <span style="color: #008000;"><b>import</b></span> <span style="color: #FF6D00;">NCUT</span>, rgb_from_tsne_3d
<span style="color: #008000;"><b>from</b></span> ncut_pytorch.backbone_text <span style="color: #008000;"><b>import</b></span> load_text_model

<span>os.environ['HF_ACCESS_TOKEN'] = </span><span style="color: #008080;">"your_huggingface_token"</span>
llama = load_text_model(<span style="color: #ab38f2;">"meta-llama/Meta-Llama-3.1-8B"</span>).cuda()
output_dict = llama(<span style="color: #808080;">"The quick white fox jumps over the lazy cat."</span>)

model_features = output_dict[<span style="color: #ab38f2;">'block'</span>][<span style="color: #008080;">31</span>].squeeze(<span style="color: #008080;">0</span>)  <span style="color: #008080;"># 32nd block output</span>
token_texts = output_dict[<span style="color: #ab38f2;">'token_texts'</span>]
eigvectors, eigvalues = <span style="color: #FF6D00;">NCUT</span>(num_eig=<span style="color: #008080;">5</span>, device=<span style="color: #ab38f2;">'cuda:0'</span>).fit_transform(model_features)
tsne_x3d, tsne_rgb = rgb_from_tsne_3d(eigvectors, device=<span style="color: #ab38f2;">'cuda:0'</span>)
<span style="color: #008080;"># eigvectors.shape[0] == tsne_rgb.shape[0] == len(token_texts)</span>
    </code></pre>
</div>



---

## Why NCUT

Normalized cut offers two advantages:

1. soft-cluster assignments as eigenvectors

2. hierarchical clustering by varying the number of eigenvectors

<div  style="text-align: center;">
<video width="80%" controls muted autoplay loop>
  <source src="./images/n_eigvecs.mp4" type="video/mp4">
</video>
<p>Video: Heatmap is cosine distance of eigenvectors, w.r.t the mouse pixel (blue point).</br>
Reduce `n_eig` hierarchical grow the object heatmap</br>
try it at <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">🤗HuggingFace Demo</a> (switch to tab "PlayGround")
</div>

Please see [NCUT and t-SNE/UMAP](compare.md) for a comparison over common PCA, t-SNE, UMAP.


---

> paper in prep, Yang 2024
>
> AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*,2024
> 
> Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
