# %%
from ncut_pytorch.backbone import load_model
# %%
sdv14 = load_model('Diffusion(CompVis/stable-diffusion-v1-4)')
# %%
sdv14.pipe.unet
# %%
