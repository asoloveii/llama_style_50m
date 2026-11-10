import os
import glob
import bisect
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


def fineweb_get_data(shard_dir, 
                     context_size=1024,
                     test_size=0.05):
    ds = FineWebDataset(shard_dir, context_size)
    
    # calculate split point based on total sample count
    val_size = int(len(ds) * test_size)
    train_size = len(ds) - val_size
    
    train_ds = torch.utils.data.Subset(ds, range(0, train_size))
    val_ds = torch.utils.data.Subset(ds, range(train_size, len(ds)))
    return train_ds, val_ds


class FineWebDataset(Dataset):

    def __init__(self, shard_dir: str, context_size: int = 1024):
        self.context_size = context_size

        # find and sort all .npy files in the directory
        shard_pattern = os.path.join(shard_dir, "*.npy")
        self.shard_paths = sorted(glob.glob(shard_pattern))
        
        if not self.shard_paths:
            raise ValueError(f"No .npy shard files found in directory: {shard_dir}")

        # inspect shards to map global indices to specific shards and offsets
        self.shard_lengths = []
        self.cumul_lengths = [0]

        for path in self.shard_paths:
            mmap_array = np.load(path, mmap_mode="r")
            length = len(mmap_array)
            self.shard_lengths.append(length)
            self.cumul_lengths.append(self.cumul_lengths[-1] + length)

        self.total_tokens = self.cumul_lengths[-1]
        self.total_samples = max(0, (self.total_tokens - 1) // context_size)

        self._mmap_cache = {}
        self._max_cache_size = 8
    
    def __len__(self):
        return self.total_samples
    
    def _get_mmap(self, shard_idx: int) -> np.ndarray:
        if shard_idx in self._mmap_cache:
            return self._mmap_cache[shard_idx]
        
        # evict oldest entry if cache is full 
        if len(self._mmap_cache) >= self._max_cache_size:
            oldest_idx = next(iter(self._mmap_cache))
            del self._mmap_cache[oldest_idx]
            
        mmap_array = np.load(self.shard_paths[shard_idx], mmap_mode="r")
        self._mmap_cache[shard_idx] = mmap_array
        return mmap_array

    def __getitem__(self, idx: int):
        start_token_idx = idx * self.context_size
        tokens_needed = self.context_size + 1

        tokens = []
        current_token_idx = start_token_idx

        while len(tokens) < tokens_needed and current_token_idx < self.total_tokens:
            shard_idx = bisect.bisect_right(self.cumul_lengths, current_token_idx) - 1
            
            shard_start_token = self.cumul_lengths[shard_idx]
            shard_offset = current_token_idx - shard_start_token
            mmap_array = self._get_mmap(shard_idx)
            
            available_in_shard = len(mmap_array) - shard_offset
            needed = tokens_needed - len(tokens)
            chunk_to_take = min(needed, available_in_shard)

            chunk = mmap_array[shard_offset : shard_offset + chunk_to_take]
            tokens.append(chunk)

            current_token_idx += chunk_to_take

        tokens = np.concatenate(tokens)

        # strict safety padding
        if len(tokens) < tokens_needed:
            tokens = np.pad(tokens, (0, tokens_needed - len(tokens)), mode="constant")
        else:
            tokens = tokens[:tokens_needed]  

        x = torch.from_numpy(tokens[:-1].astype(np.int64))  
        y = torch.from_numpy(tokens[1:].astype(np.int64))   

        return x, y
