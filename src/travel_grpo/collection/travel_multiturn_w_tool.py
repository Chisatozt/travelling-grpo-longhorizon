"""
Preprocess the TravelGym dataset to parquet format
"""

import argparse
import os
import re
import json
from pathlib import Path
from datasets import Dataset
import numpy as np

from .._paths import REPOSITORY_ROOT

PROJECT_ROOT = REPOSITORY_ROOT


def main(wanted_num, one_choice_per_aspect):

    local_dir = PROJECT_ROOT / "data" / "grpo" / f"travel{wanted_num}_multiturn"
    if one_choice_per_aspect:
        local_dir = local_dir.with_name(local_dir.name + "_onechoice")

    data_source = PROJECT_ROOT / "environments" / "TravelGym" / "travelgym" / "data" / f"travelgym_data_{wanted_num}.json"
    with data_source.open(encoding="utf-8") as source_file:
        dataset = json.load(source_file)
    dataset_len = len(dataset)
    
    all_data = []
    for key, value in dataset.items():
        value["id"] = key
        all_data.append(value)
    
    # let the value to form a list
    dataset = list(all_data)
    np.random.shuffle(dataset)
    
    train_dataset = dataset[:int(dataset_len * 0.85)]
    test_dataset = dataset[int(dataset_len * 0.85):]
    
    print(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")
    
    # add a row to each data item that represents a unique id
    def make_map_fn(example, idx, split):
        id = example.pop("id")

        initial_description = example["initial_description"]

        different_sentence = "  * Usually you should start by performing a search, then take action to actively uncover the user's preferences or reason to provide an answer. Keep in mind that multiple travel aspects require answers, and you are allowed to recommend only one option per aspect. Therefore, before making a recommendation, ensure you have thoroughly communicated with the user to understand their preferences.\n" if one_choice_per_aspect else \
        "  * Usually you should start by performing a search, then take action to actively uncover the user's preferences or reason to provide an answer. Keep in mind that multiple travel aspects require answers, and while you may answer multiple times, each answer should include only one option ID.\n"

        data = {
            "data_source": "interact_travelgym",
            "prompt": [
                {
                    "role": "system",
                    "content":
                            "You are an agent that actively interact with a specific environment. The followings are the details of the environment and your action space.\n\n" +
                            "- Environment Description: TravelGym is an environment where you interact with both a user and a search database to fulfill a travel plan. Since the user's initial intent may be incomplete, you must proactively elicit preferences, perform searches, and make informed recommendations.\n\n" +
                            "- Action Space: You should call the tool `interact_with_env` to interact with the environment. The action should be one of the following: `search`, `action` or `answer`.\n\n" +
                            "- Action Description:\n" +
                            "  * `search`: If you choose `search`, you must issue a clear and detailed query to the database in the `content` field. Specify the travel aspect you are searching for (e.g., hotel, flight, etc.) and provide well-supported arguments for your query. Only make one focused search attempt at a time.\n" +
                            "  * `action`: If you choose `action`, you will communicate directly with the user through the message you write in the `content` field. Your goal is to obtain natural-language evidence about preferences and constraints. The environment does not shrink or pre-filter the candidate list; compare the evidence with the complete Search results yourself. Avoid vague or overly general inquiries, and focus on detailed aspects of their travel needs.\n" +
                            "  * `answer`: If you choose `answer`, you must recommend a specific option to the user in the `content` field. Please submit exactly one option ID that appeared in the Search result for the current travel aspect.\n\n" +
                            "- Important Notes:\n" +
                            "  * In each step of interaction, first write your thoughts and analysis between `<think>` and `</think>` to carefully decide your next step. Only after providing this reasoning should you call the `interact_with_env` tool to interact with the environment. Always present your reasoning before making the tool call.\n" +
                            "  * The total number of rounds that you can interact with the environment is limited. You should smartly balance the number of rounds that you search, take action, or provide answer, so that you can fulfill the user's travel preferences in the most efficient way.\n" +
                            # Ensure the prompt reflects the one_choice_per_aspect setting
                            different_sentence +
                            "  * For each travel aspect, Search returns the complete candidate list. Action is evidence gathering only: the user replies in natural language and never exposes a preference ID or a pre-filtered subset. You must perform the final implicit comparison and choose the visible option ID.\n" +
                            "  * Be bold, creative and smart in your interaction with the environment! Let's begin!",
                },
                {
                    "role": "user",
                    "content": (
                            f"{initial_description}\nAlso my budget is limited so as long as my preferences are satisfied, I would also like to choose the cheapest option for each.\n"
                    ),
                },
            ],
            "ability": "interaction",
            # Correct/best IDs stay in the private environment ledger. Keeping
            # this field empty prevents rollout parquet files from becoming a
            # label side channel to the Actor.
            "reward_model": {"style": "terminal", "ground_truth": "", "env_name": "TravelGym", "id": id},
            "extra_info": {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "TravelGym", "id": id},
                    },
                },
            },
        }
        return data

    train_dataset = [make_map_fn(example, idx, "train") for idx, example in enumerate(train_dataset)]
    test_dataset = [make_map_fn(example, idx, "test") for idx, example in enumerate(test_dataset)]
    
    # Make it into Dataset with features
    train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)
    
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(str(local_dir / "train.parquet"))
    test_dataset.to_parquet(str(local_dir / "test.parquet"))

if __name__ == "__main__":
    for wanted_num in ["2222", "444", "334", "333", "233", "44", "22", "33"]:
        for one_choice_per_aspect in [True]:
            np.random.seed(42)
            print(f"Processing wanted_num: {wanted_num}, one_choice_per_aspect: {one_choice_per_aspect}")
            main(wanted_num, one_choice_per_aspect)
