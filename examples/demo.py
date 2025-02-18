"""
Demo script showing how to use the ReduxPipelineWrapper.
Make sure you have an appropriate reduxprior pipeline checkpoint.
"""

import torch
from PIL import Image
from diffusers import FluxPriorReduxPipeline
from reduxcache import CacheManager, ReduxPipelineWrapper, DEFAULT_CACHE_DIR, DEFAULT_CACHE_EXPIRY

def main():
    # Initialize the reduxprior pipeline (adjust the model id as needed)
    pipeline = FluxPriorReduxPipeline.from_pretrained("your-redux-model-id", torch_dtype=torch.float16)
    pipeline.to("cuda")
    
    # Initialize the cache manager with default settings.
    cache_manager = CacheManager(cache_dir=DEFAULT_CACHE_DIR, expiry=DEFAULT_CACHE_EXPIRY)
    
    # Wrap the reduxprior pipeline with our caching wrapper.
    redux_wrapper = ReduxPipelineWrapper(pipeline, cache_manager)
    
    # Load an image and define a prompt.
    # You can load an image from file:
    image = Image.open("path/to/your/image.png").convert("RGB")
    prompt = "A red cat playing with a ball"
    
    # Retrieve embeddings (this will compute them if not cached)
    embeddings = redux_wrapper.get_embeddings(image=image, prompt=prompt)
    
    # Print shapes for verification
    print("Prompt Embeds Shape:", embeddings["prompt_embeds"].shape)
    print("Pooled Prompt Embeds Shape:", embeddings["pooled_prompt_embeds"].shape)
    
    # Now you can feed these embeddings to your downstream image generation pipeline.
    
if __name__ == "__main__":
    main()
