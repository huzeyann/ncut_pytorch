# %%

from ncut_pytorch.predictor.jafar_predictor import NcutJafarPredictor
from ncut_pytorch.predictor.dino_predictor import NcutDinov3Predictor
from PIL import Image

# predictor = NcutJafarPredictor(model_name="siglip2_b", batch_size=1, input_resolution=(384, 384), output_resolution=(384, 384))
predictor = NcutDinov3Predictor(model_cfg="dinov3_vitl16")
predictor = predictor.to('cuda')

images = [Image.open(f"images/view_{i}.jpg") for i in range(4)]
predictor.set_images(images)

# %%

image = predictor.summary(n_segments=[10, 25, 50, 100], draw_border=True)
display(image)



# %%
