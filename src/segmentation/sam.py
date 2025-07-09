import cv2
from exporch import get_available_device
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import requests
from segment_anything import sam_model_registry, SamPredictor


def segment(image_path: Path, output_dir: Path, model_dir: Path, verbose: bool=True, clean: bool=True):
    # Loading the image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if verbose:
        print(f"The shape of the image is ({image.shape[0]}, {image.shape[1]})")

    # Defining some points of the image that will help SAM in the segmentation process
    center = (image.shape[1] // 2, image.shape[0] // 2)
    first_vertex = (10, 10)
    second_vertex = (image.shape[1] - 10, 10)
    third_vertex = (image.shape[1] - 10, image.shape[0] - 10)
    fourth_vertex = (10, image.shape[0] - 10)

    # Showing the image
    if verbose:
        plt.figure()
        plt.imshow(image)
        plt.scatter(*zip(*[center, first_vertex, second_vertex, third_vertex, fourth_vertex]),
                    c="Red", s=40, label="Points")
        plt.axis('off')
        plt.show()

    # Loading SAM model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam_path = model_dir / sam_checkpoint
    if not sam_path.exists():
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(sam_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    device = get_available_device("mps")
    sam = sam_model_registry[model_type](checkpoint=sam_path)
    sam.to(device)
    if verbose:
        print("The model is on device:", device)

    # Creating the predictor
    predictor = SamPredictor(sam)

    # Setting the image to the predictor
    predictor.set_image(image)

    # Choosing the point that will guide the segmentation
    input_point = np.array([
        [center[0], center[1]],
        [first_vertex[0], first_vertex[1]],
        [second_vertex[0], second_vertex[1]],
        [third_vertex[0], third_vertex[1]],
        [fourth_vertex[0], fourth_vertex[1]]
    ])
    input_label = np.array([1, 0, 0, 0, 0])

    # Predicting the mask (multimask_output=True returns 3 masks)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Extracting the best mask
    best_index = np.argmax(scores)
    image_mask = masks[best_index]

    # Storing the mask
    file_name = image_path.stem + "_mask.png"
    mask_path = output_dir / file_name
    Image.fromarray(image_mask.astype(np.uint8), mode="L").save(mask_path)
    # mode="L" allows to create binary images (masks)

    # Visualizing the results
    for i, mask in enumerate(masks):
        plt.figure()
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)
        plt.title(f"Mask {i + 1} - Score: {scores[i]:.3f}")
        plt.axis("off")
    plt.show()

    if clean:
        # Deleting the model
        del sam, predictor

    return mask_path