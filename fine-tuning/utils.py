import os
import numpy as np

from typing import Tuple
from task9_scorer import *

# == PATH == #
BASE_PATH = "./dataset/"

# sub task define
subtask_dict = ["1A", "1B", "1C", "2A", "2B"]
subtask_path_dict = {
    "1A": "1A.english.",
    "1B": "1B.italian.",
    "1C": "1C.spanish.",
    "2A": "2A.medical.",
    "2B": "2B.music.",
}


def load_vocabs(subtask: str) -> list:
    assert subtask in subtask_dict, Exception(f"Exist subtask : {subtask_dict}")
    # path
    vocab_dir = "vocabulary"
    file_prefix = f"{subtask_path_dict[subtask]}"
    file_path = f"vocabs/{file_prefix}{vocab_dir}.txt"
    vocab_path = os.path.join(BASE_PATH, file_path)

    # load
    vocabs = []
    with open(vocab_path, "r") as fr:
        for vocab in fr:
            vocabs.append(vocab.strip())

    return vocabs


def load_train_dataset(subtask, dir_name):
    file_prefix = f"{subtask_path_dict[subtask]}"
    data_file_path = f"{dir_name}/data/{file_prefix}training.data.txt"
    gold_file_path = f"{dir_name}/gold/{file_prefix}training.gold.txt"

    data_file_path = os.path.join(BASE_PATH, data_file_path)
    gold_file_path = os.path.join(BASE_PATH, gold_file_path)

    dataset = []
    gold_dict = {}

    with open(data_file_path, "r") as dfr, open(gold_file_path, "r") as gfr:
        for query, gold in zip(dfr, gfr):
            query, gold = query.strip(), gold.strip()

            query, types = query.split("\t")
            golds = gold.split("\t")

            gold_dict[query] = []

            for g in golds:
                gold_dict[query].append(g)
                dataset.append([types, query, g])
    return dataset, gold_dict


def load_datasets(subtask: str, repeated=1) -> Tuple[list, list, list]:
    assert subtask in subtask_dict, Exception(f"Exist subtask : {subtask_dict}")

    data_dir = f"shuffled_dataset/data_{str(repeated).zfill(2)}"

    train_dataset, gold_dict = load_train_dataset(subtask, data_dir)

    return train_dataset, gold_dict
