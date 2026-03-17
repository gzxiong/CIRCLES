import argparse
import os
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from load_data import load_dataset


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CLIP embeddings for train/test splits")
    parser.add_argument("--data", type=str, required=True, choices=["okvqa", "vizwiz", "cub", "flowers"])
    parser.add_argument("--clip_model", type=str, default="laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    parser.add_argument("--save_root", type=str, default=str(REPO_ROOT / "embeddings"))
    parser.add_argument("--text_batch_size", type=int, default=128)
    parser.add_argument("--image_batch_size", type=int, default=64)
    args = parser.parse_args()

    clip_model = CLIPModel.from_pretrained(args.clip_model, device_map="auto")
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    save_dir = os.path.join(args.save_root, args.data)
    os.makedirs(save_dir, exist_ok=True)

    for split in ["train", "test"]:
        print(f"Loading {args.data}/{split}")
        dataset = load_dataset(args.data, split=split)

        text_save = os.path.join(save_dir, f"{split}_text_embeddings.pt")
        image_save = os.path.join(save_dir, f"{split}_image_embeddings.pt")

        print(f"Computing text embeddings -> {text_save}")
        text_embeddings = dataset.get_text_embeddings(
            clip_model,
            clip_processor,
            batch_size=args.text_batch_size,
            save_path=text_save,
        )
        torch.save(text_embeddings, text_save)

        print(f"Computing image embeddings -> {image_save}")
        image_embeddings = dataset.get_image_embeddings(
            clip_model,
            clip_processor,
            batch_size=args.image_batch_size,
            save_path=image_save,
        )
        torch.save(image_embeddings, image_save)

    print("Done.")


if __name__ == "__main__":
    main()
