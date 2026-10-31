import os
import argparse

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def load_fineweb(total_tokens: int, shard_size: int, output_dir: str = "./datasets/fineweb"):
    """
    Downloads a subset of FineWebEDU dataset via HF streaing, tokenizes using Llama(1-2) 
    32k vocab tokenizer and then serializes tokens into .npy shard files.
    """
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    shard_index = 0
    current_shard = np.empty(shard_size, dtype=np.uint16)
    token_count = 0
    total_downloaded = 0

    for sample in dataset:
        if total_downloaded >= total_tokens:
            break

        text = sample.get("text", "")
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_id)
        tokens_np = np.array(tokens, dtype=np.uint16)

        if len(tokens_np) == 0:
            continue

        remaining_space = shard_size - token_count
        if len(tokens_np) < remaining_space:
            current_shard[token_count:token_count+len(tokens_np)] = tokens_np
            token_count += len(tokens_np)
            total_downloaded += len(tokens_np)
        else:
            current_shard[token_count:token_count+remaining_space] = tokens_np[:remaining_space]

            shard_path = os.path.join(output_dir, f"shard_{shard_index:06d}.npy")
            np.save(shard_path, current_shard)
            print(f"Saved shard {shard_index} -> {shard_path}")

            shard_index += 1
            total_downloaded += remaining_space

            leftover = tokens_np[remaining_space:]
            while len(leftover) >= shard_size:
                shard_path = os.path.join(output_dir, f"shard_{shard_index:06d}.npy")
                np.save(shard_path, leftover[:shard_size])
                print("Saved shard {shard_index} -> {shard_path}")
                shard_index += 1
                total_downloaded += shard_size
                leftover = leftover[shard_size:]

            current_shard = np.empty(shard_size, dtype=np.uint16)
            token_count = len(leftover)
            if token_count > 0:
                current_shard[:token_count] = leftover
                total_downloaded += token_count

        if total_downloaded >= total_tokens:
            break


    print(f"Download and serialization complete! Total tokens: {total_downloaded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FineWebEDU loading into .npy binary shards")

    parser.add_argument("--total_tokens", type=int, default=5_000_000, help="Total number of tokens to download.")
    parser.add_argument("--shard_size", type=int, default=1_000_000, help="Number of tokens per shard")
    parser.add_argument("--output_dir", type=str, default="./datasets/fineweb", help="Outputer folder")

    args = parser.parse_args()

    load_fineweb(
        total_tokens=args.total_tokens,
        shard_size=args.shard_size,
        output_dir=args.output_dir
    )

