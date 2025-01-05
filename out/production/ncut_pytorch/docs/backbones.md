#  Feature Extraction from Backbone Models

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

## Loading models

```py linenums="1"
from ncut_pytorch.backbone import load_model  # image models
model = load_model(model_name='Diffusion(CompVis/stable-diffusion-v1-4)')

from ncut_pytorch.backbone_text import load_text_model  # text models
gpt2 = load_text_model("gpt2")
```

#### Install Extra Dependency

Some models require installing extra dependency, an error message will be printed if the dependency is not installed, please follow the error message to install the dependency.

```shell
>>> model = load_model("SAM(sam_vit_b)")
Error: Please install segment_anything from https://github.com/facebookresearch/segment-anything.git
pip install git+ttps://github.com/facebookresearch/segment-anything.git

>>> model = load_model("Diffusion(stabilityai/stable-diffusion-2)")
Error: Please install diffusers to use this class.
pip install diffusers
```

#### `HF_ACCESS_TOKEN` HuggingFace Access Token

Some models requires setup [HuggingFace access token](https://huggingface.co/docs/hub/security-tokens) to access. You will need to 

1. request access at their HuggingFace repository (listed below). 
 
2. put your access token to environment variable `HF_ACCESS_TOKEN`

```python
os.environ['HF_ACCESS_TOKEN'] = "your_token_here"
llama = load_model("meta-llama/Meta-Llama-3.1-8B")
```

|model_name|HuggingFace repository|
|---|---|
|stabilityai/stable-diffusion-3-medium| [https://huggingface.co/stabilityai/stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) |
|meta-llama/Meta-Llama-3.1-8B| [https://huggingface.co/meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) |
|meta-llama/Meta-Llama-3-8B| [https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |


## Feature Extraction Output

Feature extraction output is a dictionary:

1. dict keys are `node_name` (attention layer, mlp layer, etc.)

2. dict values are a list of features, each element in the list is for each layer of the model.

```py
# out_dict = {node_name: List[layer_0, layer_1, ...]}
```

#### Image Model Output

```python
from ncut_pytorch.backbone import load_model
import torch
model = load_model(model_name="SAM(sam_vit_b)").cuda()
images = torch.rand(1, 3, 1024, 1024).cuda()
out_dict = model(images)

for node_name in out_dict.keys():
    print(f"node_name: `{node_name}`, num_layers: {len(out_dict[node_name])}")
    print(f"layer_0 shape: {out_dict[node_name][0].shape}")
```

```
node_name: `attn`, num_layers: 12
layer_0 shape: torch.Size([1, 64, 64, 768])
node_name: `mlp`, num_layers: 12
layer_0 shape: torch.Size([1, 64, 64, 768])
node_name: `block`, num_layers: 12
layer_0 shape: torch.Size([1, 64, 64, 768])
```

#### Text Model Output

```python
from ncut_pytorch.backbone_text import load_text_model
import torch
model = load_text_model(model_name="gpt2").cuda()
out_dict = model("I know this sky loves you")

for node_name in out_dict.keys():
    if isinstance(out_dict[node_name][0], torch.Tensor):
        print(f"node_name: `{node_name}`, num_layers: {len(out_dict[node_name])}")
        print(f"layer_0 shape: {out_dict[node_name][0].shape}")
    else:
        print(f"token_texts: {out_dict[node_name]}")
```

```
node_name: `attn`, num_layers: 12
layer_0 shape: torch.Size([1, 6, 768])
node_name: `mlp`, num_layers: 12
layer_0 shape: torch.Size([1, 6, 768])
node_name: `block`, num_layers: 12
layer_0 shape: torch.Size([1, 6, 768])
token_texts: ['I', ' know', ' this', ' sky', ' loves', ' you']
```