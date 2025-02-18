import hashlib
from typing import Union, List
from PIL import Image
import io
import numpy as np

def hash_prompt(prompt: Union[str, List[str]]) -> str:
    """
    Generate a SHA256 hash for the prompt text.
    
    :param prompt: A single prompt string or a list of prompt strings.
    :return: A hexadecimal hash string.
    """
    if isinstance(prompt, list):
        prompt_str = "||".join(prompt)
    else:
        prompt_str = prompt
    prompt_str = prompt_str.strip().lower()
    return hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()

def hash_image(image: Union[Image.Image, np.ndarray]) -> str:
    """
    Generate a SHA256 hash for the image.
    
    :param image: A PIL Image or a numpy array representing the image.
    :return: A hexadecimal hash string.
    """
    if isinstance(image, np.ndarray):
        image_bytes = image.tobytes()
    elif isinstance(image, Image.Image):
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
    else:
        raise ValueError("Unsupported image type for hashing.")
    return hashlib.sha256(image_bytes).hexdigest()

def combine_hashes(image_hash: str, prompt_hash: str) -> str:
    """
    Combine the image and prompt hashes to create a unique key.
    
    :param image_hash: Hash string of the image.
    :param prompt_hash: Hash string of the prompt.
    :return: Combined SHA256 hash as a hexadecimal string.
    """
    combined = image_hash + prompt_hash
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
