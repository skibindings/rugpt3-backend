#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
import os

import argparse
import logging

import numpy as np
import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def set_seed(args):
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["n_gpu"] > 0:
        torch.cuda.manual_seed_all(args["seed"])

class ruGPTModel:
    # init method or constructor
    def __init__(self):
        self.args = {"prompt": "Это тест",
                     "length": 20,
                     "stop_token": "</s>",
                     "temperature": 1.0,
                     "repetition_penalty": 1.0,
                     "k": 0,
                     "p": 0.9,
                     "seed": 42,
                     "no_cuda": True,
                     "num_return_sequences": 1}


        self.args["device"] = torch.device("cuda" if torch.cuda.is_available() and not self.args["no_cuda"] else "cpu")
        self.args["n_gpu"] = 0 if self.args["no_cuda"] else torch.cuda.device_count()

        set_seed(self.args)

        self.tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
        self.model.to(self.args["device"])

        self.args["length"] = adjust_length_to_model(self.args["length"], max_sequence_length=self.model.config.max_position_embeddings)
        logger.info(self.args)

    def set_hyper_params(self, params):
      self.args["k"]=params["k"]
      self.args["p"]=params["p"]
      self.args["temperature"]=params["temperature"]
      self.args["repetition_penalty"]=params["repetition_penalty"]
      self.args["length"]=params["length"]
      self.args["num_return_sequences"]=params["num_return_sequences"]
      self.args["seed"]=params["seed"]
      set_seed(self.args)

    def inference(self, context):
        generated_sequences = []
        prompt_text = context

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.args["device"])

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=self.args["length"] + len(encoded_prompt[0]),
            temperature=self.args["temperature"],
            top_k=self.args["k"],
            top_p=self.args["p"],
            repetition_penalty=self.args["repetition_penalty"],
            do_sample=True,
            num_return_sequences=self.args["num_return_sequences"],
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print("ruGPT:".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(self.args["stop_token"]) if self.args["stop_token"] else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                    prompt_text + text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
            )

            generated_sequences.append(total_sequence)
            # os.system('clear')
            print(total_sequence)
        prompt_text = ""
        return generated_sequences
