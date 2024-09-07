

<div style="text-align: center;">
  <img src="./images/ncut.svg" alt="NCUT" style="width: 80%; filter: brightness(60%) grayscale(100%);"/>
</div>


<div style="display: flex; justify-content: center; margin-top: 20px;">

<a href="https://github.com/huzeyann/ncut_pytorch" target="_blank" style="width: 30%; text-align: center; background-color: #007BFF; color: white; padding: 10px; border-radius: 5px; margin-right: 5%;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="width: 24px; height: 24px; vertical-align: middle;"/> GitHub
</a>

<a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank" style="width: 30%; text-align: center; background-color: #FF5733; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
🤗 HuggingFace Demo
</a>

</div>

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
<p>Video: NCUT applied to image encoder features from Segment Anything Model.
<a href="./gallery_sam_video">code</a>
</p>
</div>




[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gllutIdACcEHtJ81n_tGVNgR6fTupV46): A demo software for one-point prompting segmentation and pseudo-labeling.

<div  style="text-align: center;">
<video width="90%" controls muted autoplay loop>
  <source src="./images/demo_heatmap.mp4" type="video/mp4">
</video>
<p>Video: Heatmap is cosine distance of eigenvectors, w.r.t the mouse pointer.
<a href="./alignedcut_vs_ncut">details</a>
</p>
</div>


Please visit our <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">🤗HuggingFace Demo</a>. Play around models and parameters.

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.42.0/gradio.js"
></script>

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

## Installation & Quick Start

`ncut-pytorch` is available via `pip`, our package is based on PyTorch, please [Install PyTorch](https://pytorch.org/get-started/locally/) first

<div style="text-align:">
    <pre><code class="language-shell">pip install ncut-pytorch</code></pre>
</div>




<details>
<summary>

How to install PyTorch (click to expand):

</summary>

Install PyTorch by pip (for CPU only) or conda (for GPU)

<div style="text-align:">
<pre><code class="language-shell">
# for cpu only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# for gpu
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
</code></pre>
</div>

</details>

If you running into trouble when installing `ncut-pytorch`, please see [Install Trouble Shooting](touble_shooting.md)

---

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

Please see [NCUT and t-SNE/UMAP](compare.md) for a comparison over common PCA, t-SNE, UMAP.


---

> paper in prep, Yang 2024
>
> AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*,2024
> 
> Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
