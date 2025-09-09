# Copyright 2025 POLAR Team and/or its affiliates
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

import argparse
import os
import re

from datasets import load_dataset


def parse_dialogue(text: str):
    # Fixed pattern to correctly handle content with newlines
    # Uses \Z instead of $ for proper end-of-string matching
    pattern = r'^(Human|Assistant):\s*(.*?)(?=\n\n(?:Human|Assistant):|\Z)'
    matches = re.finditer(pattern, text, flags=re.MULTILINE | re.DOTALL)
    return [{"role": m.group(1).lower(), "content": m.group(2).strip()} for m in matches]


def generate_dataset(local_dir="~/data/full_hh_rlhf"):
    dataset = load_dataset("Anthropic/hh-rlhf")
    train_dataset = dataset["train"]

    data_source = "Anthropic/hh-rlhf"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            chosen = example.pop("chosen")
            example.pop("rejected")

            dialogs = parse_dialogue(chosen)
            dialogs = [{"role": "user", "content": d["content"]} if d["role"] == "human" else d for d in dialogs]

            data = {
                "data_source": data_source,
                "prompt": dialogs[:-1],
                "ability": "alignment",
                "reward_model": {
                    "style": "polar",
                    "ground_truth": dialogs[-1:],
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "ability": "alignment",
                    "prompt": dialogs[:-1],
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    local_dir = os.path.expanduser(local_dir)
    local_path = os.path.join(local_dir, "train.parquet")
    train_dataset.to_parquet(local_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="~/data/full_hh_rlhf")
    args = parser.parse_args()

    generate_dataset(args.local_dir)
