import os
import pickle
import time
from typing import Optional, Tuple

import torch

class CacheEntry:
    def __init__(self, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor, timestamp: float):
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.timestamp = timestamp

class CacheManager:
    """
    A simple cache manager that stores embeddings in memory and optionally persists them on disk.
    """
    def __init__(self, cache_dir: Optional[str] = None, expiry: Optional[float] = None):
        """
        :param cache_dir: Optional directory for persisting cache entries (if None, only in-memory caching is used).
        :param expiry: Expiry time in seconds; cache entries older than this will be considered expired.
        """
        self.cache = {}  # In-memory cache: key -> CacheEntry
        self.cache_dir = cache_dir
        self.expiry = expiry
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_disk_cache()

    def _get_disk_path(self, key: str) -> str:
        """Return the full file path for a given cache key."""
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _load_disk_cache(self):
        """Load all cache entries from the cache directory."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".pkl"):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, "rb") as f:
                        entry = pickle.load(f)
                    self.cache[filename[:-4]] = entry
                except Exception as e:
                    print(f"Error loading cache file {filename}: {e}")

    def _save_to_disk(self, key: str, entry: CacheEntry):
        """Persist a cache entry to disk."""
        if not self.cache_dir:
            return
        path = self._get_disk_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(entry, f)
        except Exception as e:
            print(f"Error saving cache file {path}: {e}")

    def get_cache(self, key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve cached embeddings if available and not expired.
        
        :param key: The unique key for the input.
        :return: Tuple (prompt_embeds, pooled_prompt_embeds) if available; otherwise None.
        """
        entry = self.cache.get(key)
        if entry:
            if self.expiry and (time.time() - entry.timestamp > self.expiry):
                self.delete_cache(key)
                return None
            return entry.prompt_embeds, entry.pooled_prompt_embeds
        return None

    def set_cache(self, key: str, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor) -> None:
        """
        Store embeddings in the cache.
        
        :param key: Unique key for the inputs.
        :param prompt_embeds: Detailed token-level embeddings.
        :param pooled_prompt_embeds: Pooled (summary) embeddings.
        """
        entry = CacheEntry(prompt_embeds, pooled_prompt_embeds, time.time())
        self.cache[key] = entry
        self._save_to_disk(key, entry)

    def flush_cache(self) -> None:
        """Clear the entire cache (both in memory and on disk, if applicable)."""
        self.cache = {}
        if self.cache_dir:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".pkl"):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except Exception as e:
                        print(f"Error deleting cache file {filename}: {e}")

    def delete_cache(self, key: str) -> None:
        """Delete a specific cache entry."""
        if key in self.cache:
            del self.cache[key]
        if self.cache_dir:
            path = self._get_disk_path(key)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error deleting cache file {path}: {e}")
