import os
import argparse

from datasets import load_dataset


def load_alpaca(output_dir: str = "./datasets/alpaca"):
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    arrow_path = os.path.join(output_dir, "train.arrow")

    dataset.save_to_disk(output_dir)
    print(f"Dataset successfully saved at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Alpaca SFT dataset")

    parser.add_argument("--output_dir", type=str, default="./datasets/alpaca", help="Output directory")

    args = parser.parse_args()

    load_alpaca(args.output_dir)

