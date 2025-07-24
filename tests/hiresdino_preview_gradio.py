# %%
import torch
from ncut_pytorch.predictor import NcutPredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

predictor = NcutPredictor(backbone='dino_512',
                          n_eig_hierarchy=[5, 10, 20, 40, 80])

default_images = ['/images/image_0.jpg', '/images/image_1.jpg', '/images/guitar_ego.jpg']

images = [Image.open(image_path) for image_path in default_images]
predictor.set_images(images)

# %%
heatmaps, masks = predictor.preview([500, 500], 0)

def blend_mask(img, mask, alpha=0.8):
    mask = mask.resize((img.width, img.height), resample=Image.Resampling.NEAREST)
    img = np.array(img).astype(np.float32)
    mask = np.array(mask).astype(np.float32)
    blend = img * (1 - alpha) + mask * alpha
    h, w = blend.shape[:2]
    blend = blend.reshape(h*w, 3)
    mask = mask.reshape(h*w, 3)
    img = img.reshape(h*w, 3)
    blend[mask[:, 0] == 255] = img[mask[:, 0] == 255]
    blend = blend.reshape(h, w, 3)
    blend = blend.astype(np.uint8)
    return Image.fromarray(blend)

def plot_masks(images, masks):
    color_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    color_values = plt.get_cmap('Reds')(color_values)
    color_values = (color_values * 255).astype(np.uint8)

    blend_images = []
    for idx, (img, mask_set) in enumerate(zip(images, masks)):
        mask = np.ones((*mask_set[0].shape, 3)) * 255
        mask = mask.astype(np.uint8)
        for i in range(len(color_values)):
            mask[masks[i][idx]] = color_values[i][:3]
        mask_img = Image.fromarray(mask)
        blend = blend_mask(img, mask_img, alpha=0.8)
        blend_images.append(blend)
    return blend_images

blend_images = plot_masks(images, masks)

# %%
def make_click_fn(img_idx):
    def click_fn(evt: gr.SelectData):
        click_xy = evt.index
        heatmaps, masks = predictor.preview(click_xy, img_idx)
        global images
        blend_images = plot_masks(images, masks)
        return gr.update(value=blend_images[0]), gr.update(value=blend_images[1]), gr.update(value=blend_images[2])
    return click_fn

def extract_features(img0, img1, img2):
    gr.Info("Extracting features...")
    global images, predictor
    images = [img0, img1, img2]
    predictor.set_images(images)
    gr.Info("Features extracted")

# %%
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            img0 = gr.Image(blend_images[0], interactive=True, type='pil')
        with gr.Column():
            img1 = gr.Image(blend_images[1], interactive=True, type='pil') 
        with gr.Column():
            img2 = gr.Image(blend_images[2], interactive=True, type='pil')

        img0.select(make_click_fn(0), outputs=[img0, img1, img2])
        img1.select(make_click_fn(1), outputs=[img0, img1, img2])
        img2.select(make_click_fn(2), outputs=[img0, img1, img2])
    
    gr.Markdown("### Click on the image to preview the hierarchy of the features (dino_vitb8)")
    with gr.Row():
        btn = gr.Button("🔴 Extract Features", variant="primary", size='lg')
        btn.click(extract_features, inputs=[img0, img1, img2], outputs=[])
    with gr.Row():
        clr_btn = gr.Button("🗑️ Clear Images", variant="secondary")
        clr_btn.click(lambda: [gr.update(value=None), gr.update(value=None), gr.update(value=None)], 
                      outputs=[img0, img1, img2])
        default_btn = gr.Button("🔄 Default Images", variant="secondary")
        default_btn.click(lambda: [gr.update(value=Image.open(default_images[0])), 
                                  gr.update(value=Image.open(default_images[1])), 
                                  gr.update(value=Image.open(default_images[2]))], 
                          outputs=[img0, img1, img2])

    
demo.launch(share=True)
# %%