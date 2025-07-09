import cv2
from diffusers import StableDiffusionInpaintPipeline
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List
import torch


models_input_size = {
    "runwayml/stable-diffusion-inpainting": (512, 512),
    "stabilityai/stable-diffusion-2-inpainting": (768, 768),
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": (1024, 1024)
}


def resize_and_pad(image, target_size=(512, 512), fill_color=(0, 0, 0)):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.copy()  # Avoid modifying the original image

    # Resize while preserving aspect ratio
    image.thumbnail(target_size, Image.LANCZOS)
    pasted_image_size = image.size

    # Create padded canvas and paste resized image
    new_image = Image.new("RGB", target_size, fill_color)
    paste_position = (
        (target_size[0] - pasted_image_size[0]) // 2,
        (target_size[1] - pasted_image_size[1]) // 2,
    )
    new_image.paste(image, paste_position)

    return new_image, paste_position, pasted_image_size


def remove_padding(padded_image, paste_position, content_size):
    left, top = paste_position
    right = left + content_size[0]
    bottom = top + content_size[1]
    # Cropping the image from the padded one
    return padded_image.crop((left, top, right, bottom))


def inpaint(
        original_image_path: Path,
        mask_path: Path,
        output_dir: Path,
        inpainting_model_id: str = "runwayml/stable-diffusion-inpainting",
        guidance_scale: float = 10.,
        trials: int = 4,
        verbose: bool = True
) -> List[Path]:
    # Loading the original image and the mask
    original_image = cv2.imread(str(original_image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Formatting the mask as the diffusion model wants
    diffusion_mask = 1 - image_mask
    diffusion_mask = Image.fromarray((diffusion_mask * 255).astype(np.uint8))

    # Resizing the image and the mask
    target_size = models_input_size[inpainting_model_id]
    prepared_image, paste_position, pasted_image_size = resize_and_pad(original_image, target_size)
    prepared_mask, _, _ = resize_and_pad(diffusion_mask, target_size, fill_color=(255, 255, 255))

    # Showing the padded image and mask
    if verbose:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(prepared_image)
        ax[0].set_title("Padded Image")
        ax[0].axis("off")
        ax[1].imshow(prepared_mask)
        ax[1].set_title("Padded Mask")
        ax[1].axis("off")

    # Creating the inpainting diffusion pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpainting_model_id, torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_attention_slicing()
    # pipe.enable_xformers_memory_efficient_attention()

    inpainted_images_paths = []
    rows = int(np.ceil(trials / 2))
    fig, axs = plt.subplots(rows, 2, figsize=(10, 5 * rows))
    axs = axs.flatten()

    # Applying the pipeline to the image
    inpainted_extended_images = pipe(
        prompt="A clean white studio background with soft natural shadow under the car",
        image=prepared_image,
        mask_image=prepared_mask,
        num_images_per_prompt=trials,
        guidance_scale=guidance_scale
    ).images

    for i in range(trials):
        inpainted_extended_image = inpainted_extended_images[i]
        # Removing the padding
        inpainted_image = remove_padding(inpainted_extended_image, paste_position, pasted_image_size)

        # Storing the inpainted image with padding
        file_name = original_image_path.stem + f"_inpainted_extended_{i}.png"
        inpainted_extended_path = output_dir / file_name
        inpainted_extended_image.save(inpainted_extended_path)

        # Storing the inpainted image
        file_name = original_image_path.stem + f"_inpainted_{i}.png"
        inpainted_path = output_dir / file_name
        inpainted_image.save(inpainted_path)
        inpainted_images_paths.append(inpainted_path)

        # Showing the created image
        axs[i].imshow(inpainted_image)
        axs[i].set_title(f"Inpainted Image {i}")
        axs[i].axis("off")

    plt.tight_layout()
    if verbose:
        plt.show()

    return inpainted_images_paths

if __name__ == "__main__":
    inpaint(Path("/Users/enricosimionato/Desktop/Background Editor/processing/intermeds/tesla_mask.png"), Path("/Users/enricosimionato/Desktop/Background Editor/processing/inputs/tesla.jpg"), Path("processing/outputs"))