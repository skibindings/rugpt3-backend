# Simple backend for ruGPT3Large based on BaseHTTPRequestHandler

## Required libraries
- transformers 2.8.0
- torch 1.7.0

## Manual
* Download model weights and configs form [here](https://huggingface.co/sberbank-ai/rugpt3large_based_on_gpt2/tree/main) and put them in _rugpt3_model_large_ folder
* Backend handles POST request with context string and returns ruGPT3Large generated sequence based on it
