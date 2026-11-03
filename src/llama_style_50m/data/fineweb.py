import os
import glob
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def fineweb_get_data(shard_dir, 
                     context_size=1024,
                     test_size=0.05):
    ds = FineWebDataset(shard_dir, context_size)
    train_ds, val_ds = random_split(ds, [1-test_size, test_size])
    return train_ds, val_ds


class FineWebDataset(Dataset):

    def __init__(self, shard_dir: str, context_size: int = 1024):
        self.context_size = context_size

        # find and sort all .npy files in the directory
        shard_pattern = os.path.join(shard_dir, "*.npy")
        self.shard_paths = sorted(glob.glob(shard_pattern))
        
        if not self.shard_paths:
            raise ValueError(f"No .npy shrad files found in directory: {shard_dir}")

        # inspect shard to map global indices to specific shards and offsets
        self.shard_lengths = []
        self.cumul_lengths = [0]

        for path in self.shard_paths:
            mmap_array = np.load(path, mmap_mode="r")
            length = len(mmap_array)
            self.shard_lengths.append(length)
            self.cumul_lengths.append(self.cumul_lengths[-1] + length)

        self.total_tokens = self.cumul_lengths[-1]

        # extract max (total_tokens-1) // context_size
        self.total_samples = max(0, (self.total_tokens-1) // context_size)

        self._mmap_cache = {}
    
    def __len__(self):
        return self.total_samples
    
    @lru_cache(maxsize=8)
    def _get_mmap(self, shard_idx: int) -> np.ndarray:
        return np.load(self.shard_paths[shard_idx], mmap_mode="r")

    def __getitem__(self, idx: int):
        start_token_idx = idx * self.context_size
        end_token_idx = start_token_idx + self.context_size + 1

        # collect tokens, handle cases where a seq spans across two shards
        tokens = []
        tokens_needed = self.context_size + 1

        current_token_idx = start_token_idx
        while len(tokens) < tokens_needed and current_token_idx < self.total_tokens:
            shard_idx = next(i for i, cum_len in enumerate(self.cumul_lengths[1:], start=0) if current_token_idx < cum_len)
            
            shard_start_token = self.cumul_lengths[shard_idx]
            shard_offset = current_token_idx - shard_start_token
            mmap_array = self._get_mmap(shard_idx)
            # take as much as available tokens
            available_in_shard = len(mmap_array) - shard_offset
            chunk_to_take = min(tokens_needed - len(tokens), available_in_shard)

            chunk = mmap_array[shard_offset:shard_offset+chunk_to_take]
            tokens.append(chunk)

            current_token_idx += chunk_to_take

        tokens = np.concatenate(tokens)

        if len(tokens) < tokens_needed:
            tokens = np.pad(tokens, (0, tokens_needed - len(tokens)), mode="constant")

        x = torch.from_numpy(tokens[:-1].astype(np.int64))
        y = torch.from_numpy(tokens[1:].astype(np.int64))

        return x, y

