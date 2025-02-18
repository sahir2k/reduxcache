import hashlib
from typing import Union, List, Dict
from PIL import Image
import torch
import numpy as np

# Import the reduxprior pipeline from diffusers.
# (Make sure you have a compatible version of diffusers installed.)
from diffusers import FluxPriorReduxPipeline

from reduxcache.cache_manager import CacheManager
from reduxcache.redux_utils import hash_prompt, hash_image, combine_hashes

class ReduxPipelineWrapper:
    """
    A wrapper for the FluxPriorReduxPipeline that caches its output embeddings.
    """
    def __init__(self, pipeline: FluxPriorReduxPipeline, cache_manager: CacheManager):
        """
        :param pipeline: An initialized FluxPriorReduxPipeline.
        :param cache_manager: A CacheManager instance to manage cached embeddings.
        """
        self.pipeline = pipeline
        self.cache_manager = cache_manager

    def _prepare_images(self, image: Union[Image.Image, List[Image.Image]]) -> List[Image.Image]:
        """
        Ensure the input is a list of PIL Images.
        
        :param image: A single PIL Image or list of PIL Images.
        :return: List of PIL Images.
        """
        if isinstance(image, list):
            return image
        else:
            return [image]

    def get_embeddings(self, image: Union[Image.Image, List[Image.Image]],
                       prompt: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Retrieve prompt embeddings from cache if available; otherwise, compute them using the reduxprior pipeline.
        
        :param image: A single image or a list of images (PIL Image objects).
        :param prompt: A single prompt string or a list of prompt strings.
        :return: Dictionary with keys 'prompt_embeds' and 'pooled_prompt_embeds'.
        """
        images = self._prepare_images(image)
        
        # Compute hash for each image and combine them.
        image_hashes = [hash_image(img) for img in images]
        combined_image_hash = hashlib.sha256("".join(image_hashes).encode("utf-8")).hexdigest()
        
        prompt_hash_val = hash_prompt(prompt)
        cache_key = combine_hashes(combined_image_hash, prompt_hash_val)
        
        # Check if embeddings are already cached.
        cached = self.cache_manager.get_cache(cache_key)
        if cached:
            prompt_embeds, pooled_prompt_embeds = cached
            return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}
        
        # If not cached, call the reduxprior pipeline.
        # It is assumed that the reduxprior pipeline's __call__ accepts 'prompt' and 'image'
        # and returns a dictionary with keys "prompt_embeds" and "pooled_prompt_embeds".
        outputs = self.pipeline(prompt=prompt, image=images)
        prompt_embeds = outputs.get("prompt_embeds")
        pooled_prompt_embeds = outputs.get("pooled_prompt_embeds")
        
        # Cache the computed embeddings.
        self.cache_manager.set_cache(cache_key, prompt_embeds, pooled_prompt_embeds)
        
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}
