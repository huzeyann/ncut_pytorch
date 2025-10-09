# %%
from ncut_pytorch.predictor import NcutDinov3Predictor
from PIL import Image

predictor = NcutDinov3Predictor(model_cfg="dinov3_vitl16")
predictor = predictor.to('cuda')

images = [Image.open(f"images/view_{i}.jpg") for i in range(4)]
predictor.set_images(images)

image = predictor.summary(n_segments=[10, 25, 50, 100], draw_border=True)
# %%
image
# %%
