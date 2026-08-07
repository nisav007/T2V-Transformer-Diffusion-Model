import os
import json
import argparse

import torch
from tqdm import tqdm

from transformers import (
    CLIPTokenizer,
    CLIPTextModel
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to MSRVTT JSON file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save text embeddings"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    print("Loading CLIP tokenizer...")

    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    print("Loading CLIP text encoder...")

    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(args.device)

    text_encoder.eval()

    print("Loading JSON...")

    with open(args.json_path, "r") as f:
        data = json.load(f)

    if args.max_samples is not None:
        data = data[:args.max_samples]

    print(f"Total samples: {len(data)}")

    for sample in tqdm(data):

        try:

            video_name = (
                sample["video"]
                .replace(".mp4", "")
            )

            captions = sample["caption"]

            all_embeddings = []

            for caption in captions:

                inputs = tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                )

                inputs = {
                    k: v.to(args.device)
                    for k, v in inputs.items()
                }

                with torch.no_grad():

                    outputs = text_encoder(
                        **inputs
                    )

                embedding = (
                    outputs
                    .last_hidden_state
                    .cpu()
                )

                all_embeddings.append(
                    embedding
                )

            save_path = os.path.join(
                args.output_dir,
                f"{video_name}_text.pt"
            )

            torch.save(
                all_embeddings,
                save_path
            )

        except Exception as e:

            print(
                f"Failed: {sample.get('video', 'unknown')}"
            )

            print(e)

    print("Finished.")


if __name__ == "__main__":
    main()

