# %%

from ncut_pytorch.predictor.jafar_predictor import NcutJafarPredictor
from PIL import Image

predictor = NcutJafarPredictor(model_name="dinov3_l", batch_size=1)
predictor = predictor.to('cuda')

images = [Image.open(f"images/view_{i}.jpg") for i in range(4)]
predictor.set_images(images)

# %%

image = predictor.summary(n_segments=[10, 25, 50, 100], draw_border=True)
display(image)



# %%
