# %%
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def center_crop(image, crop_width, crop_height):
    width, height = image.size
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = (width + crop_width) / 2
    bottom = (height + crop_height) / 2
    return image.crop((left, top, right, bottom))


def add_overlay(image, color=(128, 128, 128), transparency=128):
    overlay = Image.new("RGBA", image.size, color + (transparency,))
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def add_text_with_border(image, text, font_size, border_size=2):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    text_width, text_height = draw.textsize(text, font=font)
    position = ((image.width - text_width) / 2, (image.height - text_height) / 2)
    
    # Draw the border (outline) by drawing the text slightly offset in different directions
    x, y = position
    for offset in range(-border_size, border_size + 1):
        draw.text((x + offset, y), text, font=font, fill=(255, 255, 255, 255))
        draw.text((x - offset, y), text, font=font, fill=(255, 255, 255, 255))
        draw.text((x, y + offset), text, font=font, fill=(255, 255, 255, 255))
        draw.text((x, y - offset), text, font=font, fill=(255, 255, 255, 255))
    
    # Draw the main text
    draw.text(position, text, font=font, fill=(0, 0, 0, 255))  # Black text with white border
    return image


def process_image(
    image_path,
    output_path,
    text,
    crop_size=(400, 400),
    font_path="arial.ttf",
    font_size=50,
):
    # Open the image
    image = Image.open(image_path)

    # Center crop the image
    cropped_image = center_crop(image, *crop_size)

    # Add the 50% transparent gray overlay
    image_with_overlay = add_overlay(cropped_image)

    # Add the title text in bold font
    final_image = add_text_with_border(image_with_overlay, text, font_size)

    # Save the final image
    final_image.save(output_path)

#%%
input_image_path = "/nfscc/ncut_pytorch/docs/images/gallery/llama3/llama3_layer_0.jpg"
output_image_path = "/nfscc/ncut_pytorch/docs/images/gallery/llama3/llama3_cover.png"
text = "Llama3"
process_image(input_image_path, output_image_path, text, crop_size=(300, 300), font_size=50)

# convert to jpg
im = Image.open(output_image_path)
im = im.convert("RGB")
im.save(output_image_path.replace(".png", ".jpg"))
output_image_path = output_image_path.replace(".png", ".jpg")
output_image_path
# %%
