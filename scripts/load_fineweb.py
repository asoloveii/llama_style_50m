import os
import argparse
import multiprocessing as mp

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def worker_process(worker_id: int, 
                   num_workers: int, 
                   total_tokens_per_worker: int, 
                   queue: mp.Queue, 
                   stop_event: mp.Event):
    try: 
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

        # stream and parition the dataset across workers using built-in sharding
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        sharded_ds = dataset.shard(num_shards=num_workers, index=worker_id)

        tokens_sent = 0
        for sample in sharded_ds:
            if stop_event.is_set() or tokens_sent >= total_tokens_per_worker:
                break

            text = sample.get("text", "")
            if not text:
                continue

            # tokenize text and append eos token
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(eos_id)
            tokens_np = np.array(tokens, dtype=np.uint16)

            if len(tokens_np) == 0:
                continue

            queue.put(tokens_np)
            tokens_sent += len(tokens_np)

    except Exception as e:
        print(f"Worker {worker_id} encountered an error: {e}")
    finally:
        queue.put(None) # signal completion for this worker


def load_fineweb(total_tokens: int, 
                 shard_size: int, 
                 output_dir: str = "./datasets/fineweb",
                 num_workers: int = None):
    """
    Downloads a subset of FineWebEDU dataset via HF streaing, tokenizes using Llama(1-2) 
    32k vocab tokenizer and then serializes tokens into .npy shard files.
    """
    os.makedirs(output_dir, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, os.cpu_count()-1)
    
    # allocate a buffer per worker to accoutn for text length variance
    tokens_per_worker = int(np.ceil(total_tokens/num_workers*1.05))
    
    queue = mp.Queue(maxsize=2000)
    stop_event = mp.Event()
    
    workers = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(worker_id, num_workers, tokens_per_worker, queue, stop_event)
        )
        p.start()
        workers.append(p)
    
    shard_index = 0
    current_shard = np.empty(shard_size, dtype=np.uint16)
    token_count = 0
    total_downloaded = 0

    active_workers = num_workers
    try:
        while active_workers > 0 and total_downloaded < total_tokens:
            try:
                item = queue.get(timeout=1.0)
            except Exception:
                continue

            if item is None:
                active_workers -= 1
                continue

            tokens_np = item
            if total_downloaded >= total_tokens:
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
                while len(leftover) >= shard_size and total_downloaded < total_tokens:
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

    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

    if token_count > 0 and total_downloaded > 0:
        shard_path = os.path.join(output_dir, f"shard_{shard_index:06d}_partial.npy")
        np.save(shard_path, current_shard[:token_count])

    print(f"Download and serialization complete! Total tokens: {total_downloaded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FineWebEDU loading into .npy binary shards")

    parser.add_argument("--total_tokens", type=int, default=5_000_000, help="Total number of tokens to download.")
    parser.add_argument("--shard_size", type=int, default=1_000_000, help="Number of tokens per shard")
    parser.add_argument("--output_dir", type=str, default="./datasets/fineweb", help="Outputer folder")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker process")

    args = parser.parse_args()

    load_fineweb(
        total_tokens=args.total_tokens,
        shard_size=args.shard_size,
        output_dir=args.output_dir,
        num_workers=args.num_workers 
    )

