# ReduxCache

ReduxCache is an open-source utility package that caches prompt embeddings produced by the FluxPriorReduxPipeline (reduxprior pipeline) in Hugging Face Diffusers. It is designed to save compute by caching both detailed token embeddings (`prompt_embeds`) and their pooled summary (`pooled_prompt_embeds`), so that repeated calls with the same image and prompt can quickly reuse cached results.

## Features

- Caches reduxprior pipeline outputs in memory and optionally on disk.
- Supports single image or image sets along with prompt(s).
- Configurable cache expiry and directory.

## Installation

Clone the repository and install via pip:

```bash
git clone https://github.com/yourusername/reduxcache.git
cd reduxcache
pip install .
