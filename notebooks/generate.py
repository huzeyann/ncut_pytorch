# %%
# ncut-pytorch==2.1.0
from ncut_pytorch.predictor import NcutDinov3Predictor
from PIL import Image

predictor = NcutDinov3Predictor(model_cfg="dinov3_vitl16")
predictor = predictor.to('cuda')
predictor.predictor.color_method = 'tsne'


images = [Image.open("images/view_0.jpg"), Image.open("images/view_1.jpg"), Image.open("images/view_2.jpg")
            , Image.open("images/view_3.jpg"), Image.open("images/view_ego.jpg"), Image.open("images/image2.jpg")]

predictor.set_images(images)



# %%
segments = predictor.generate(n_segment=20)
# %%
color = predictor.color_discrete(segments, draw_border=True)
# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for i in range(6):
    axes[0, i].imshow(images[i])
    axes[0, i].axis('off')
    axes[1, i].imshow(color[i])
    axes[1, i].axis('off')
plt.show()
# %%
image = predictor.summary(draw_border=True)
image
# %%
