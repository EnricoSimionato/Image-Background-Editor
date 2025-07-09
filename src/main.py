import argparse
from diffusion.inpainter import inpaint
from pathlib import Path
from segmentation.sam import segment
from typing import List


INPUTS_DIR = Path("processing/inputs")
INTERMEDS_DIR = Path("processing/intermeds")
OUTPUTS_DIR = Path("processing/outputs")
MODELS_DIR = Path("models")


def main(image_name: str) -> List[Path]:
    image_path = INPUTS_DIR / image_name
    mask_path = segment(image_path, INTERMEDS_DIR, MODELS_DIR)
    inpainted_image_path = inpaint(image_path, mask_path, OUTPUTS_DIR)

    return inpainted_image_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Background removal")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    main(args.image_path)
