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
import json

from datasets import Dataset
from huggingface_hub import snapshot_download

filtered_num = 0
data_source = "a-m-team/AM-DeepSeek-R1-0528-Distilled"


def read_dataset(local_dir):
    global filtered_num
    idx = 0
    for file in os.listdir(local_dir):
        if file.endswith(".jsonl"):
            with open(os.path.join(local_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    example = json.loads(line)
                    try:
                        conversations = example.pop("conversations")

                        dialogs = []
                        for item in conversations:
                            if item["from"] == "human":
                                dialogs.append({"role": "user", "content": item["value"]})
                            else:
                                if "info" in item:
                                    dialogs.append({"role": "assistant", "content": item["info"]["answer_content"]})
                                else:
                                    content = item["value"].split("<answer>")[1].split("</answer>")[0].strip()
                                    dialogs.append({"role": "assistant", "content": content})

                        assert dialogs[-1]["role"] == "assistant"
                        data = {
                            "data_source": data_source,
                            "prompt": dialogs[:-1],
                            "ability": "general",
                            "reward_model": {
                                "style": "polar",
                                "ground_truth": dialogs[-1:],
                            },
                            "extra_info": {
                                "split": "train",
                                "index": idx,
                                "ability": "general",
                                "prompt": dialogs[:-1],
                            },
                        }
                        yield data
                        idx += 1
                    except Exception as e:
                        print(f"Error processing example {idx}: {e}")
                        filtered_num += 1


def generate_dataset(local_dir="~/data/general"):

    data_dir = snapshot_download(
        repo_id="a-m-team/AM-DeepSeek-R1-0528-Distilled",
        repo_type="dataset",
        revision="main",
        local_dir="~/data/AM-DeepSeek-R1-0528-Distilled",
        local_dir_use_symlinks=False,
        allow_patterns=["*.jsonl"],
        ignore_patterns=["*.png"]
    )

    final_dataset = Dataset.from_generator(lambda: read_dataset(data_dir))
    local_dir = os.path.expanduser(local_dir)
    local_path = os.path.join(local_dir, "train.parquet")
    final_dataset.shuffle(seed=42).to_parquet(local_path)
    print(f"Filtered {filtered_num} examples due to errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="~/data/general")
    args = parser.parse_args()

    generate_dataset(args.local_dir)
