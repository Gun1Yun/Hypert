import os
import argparse
import time
import random
from copy import deepcopy
from itertools import cycle
from math import sqrt

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from model import *
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()

parser.add_argument("task", metavar="str", type=str, help="task name")
parser.add_argument("ckpt", metavar="str", type=str, help="model checkpoint")

SEED = 4242


# seed setting
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


def save_result(hypernyms, save_path):
    results = ""

    for hypernym in hypernyms:
        results += "\t".join(hypernym)
        results += "\n"

    with open(save_path, "w", encoding="UTF8") as fw:
        fw.write(results)


# BERT ==
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
type_token_dict = {"Concept": "[CON]", "Entity": "[ENT]"}
new_tokens = ["[CON]", "[ENT]"]

special_tokens_dict = {"additional_special_tokens": new_tokens}
tokenizer.add_special_tokens(special_tokens_dict)


def make_sampler(things):
    """Make generator that samples randomly from a list of things."""
    nb_things = len(things)
    shuffled_things = deepcopy(things)
    for i in cycle(range(nb_things)):
        if i == 0:
            random.shuffle(shuffled_things)
        yield shuffled_things[i]


# pos sample prob
def get_pos_sample_prob(pairs):
    pos_sample_prob = {}
    hyp_fd = {}
    for d in pairs:
        h_id = d[2]
        if h_id not in hyp_fd:
            hyp_fd[h_id] = 0
        hyp_fd[h_id] += 1
    min_freq = min(hyp_fd.values())
    for h_id, freq in hyp_fd.items():
        pos_sample_prob[h_id] = sqrt(min_freq / freq)

    return pos_sample_prob


def generate_batch(pairs, gold_dict, vocab_sampler, subsample, pos_sample_prob, batch_size, n_neg):
    random.shuffle(pairs)

    n_batch = 0
    sentences = []
    labels = []

    for pair in pairs:
        t, q, h = pair
        if subsample and random.random() >= pos_sample_prob[h]:
            continue
        n_batch += 1
        sentences.append(f"{type_token_dict[t]} {q} [SEP] {h}")
        labels.append(1)

        # make negative samples
        neg_samples = []
        while len(neg_samples) < n_neg:
            neg_vocab = next(vocab_sampler)
            if neg_vocab not in gold_dict[q] or q != neg_vocab:
                neg_samples.append(neg_vocab)

        for neg_word in neg_samples:
            sentences.append(f"{type_token_dict[t]} {q} [SEP] {neg_word}")
            labels.append(0)

        if n_batch == batch_size:
            indices = list(range(len(labels)))
            random.shuffle(indices)
            sentences = np.array(sentences)
            labels = np.array(labels)
            sentences = sentences[indices].tolist()
            labels_tensors = torch.tensor(labels[indices])

            encodings = tokenizer(
                sentences, padding="longest", truncation=True, return_tensors="pt"
            )
            ids = encodings["input_ids"]
            mask_indices = torch.where(ids == 102)
            _, c_idx = mask_indices

            # Generate masking vectors: MQ, MC
            query_mask = torch.zeros_like(ids)
            hyper_mask = torch.zeros_like(ids)
            for r, idx in enumerate(range(0, len(c_idx), 2)):
                query_mask[r, 2 : c_idx[idx]] = 1
                hyper_mask[r, c_idx[idx] + 1 : c_idx[idx + 1]] = 1

            # reset
            n_batch = 0
            sentences = []
            labels = []

            yield encodings, query_mask, hyper_mask, labels_tensors


def main():
    args = parser.parse_args()
    task = args.task
    model_path = args.ckpt

    num_of_epochs = 15
    learning_rate = 27e-7
    batch_size = 32
    n_negs = 50
    subsample = False
    freeze_bert = False

    n_repeat = 10
    task_dict = {"1A": "1A.english", "2A": "2A.medical", "2B": "2B.music"}

    # set path
    base_model_dir = f"./outputs/{task}model"
    if subsample:
        base_model_dir += "_subsample"
    if freeze_bert:
        base_model_dir += "_freeze"

    if not os.path.exists(base_model_dir):
        os.makedirs(base_model_dir)

    vocabs = load_vocabs(task)
    vocab_sampler = make_sampler(vocabs)

    # repeat
    for n in range(1, 1 + n_repeat):
        model_save_path = os.path.join(base_model_dir, f"{str(n).zfill(2)}_model.pt")
        log_path = os.path.join(base_model_dir, f"{str(n).zfill(2)}_model_log.txt")

        # == load dataset == #
        # load n-th dataset
        train_dataset, gold_dict = load_datasets(task, n)
        pos_sample_prob = get_pos_sample_prob(train_dataset)  # get subsampling prob

        model = HypertModel(model_path)
        model.bert.resize_token_embeddings(len(tokenizer))  # for BERT, add new toknes
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
        model.cuda()

        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        print("Initialized optimizer.")

        loss_fn = nn.BCELoss()
        print("Initialized loss function.")

        loss_history = []
        check_point_headers = ["Epoch", "Loss", "Dev MAP", "Times"]

        with open(log_path, "w") as fw:
            fw.write("\t".join(check_point_headers) + "\n")

        evaluator = Evaluator(model, tokenizer, 32768, task, 15)

        # load trial set
        x_dev_path = f"./dataset/shuffled_dataset/data_{str(n).zfill(2)}/data/{task_dict[task]}.trial.data.txt"
        y_dev_path = f"./dataset/shuffled_dataset/data_{str(n).zfill(2)}/gold/{task_dict[task]}.trial.gold.txt"
        trial_pairs = {}
        with open(x_dev_path, "r", encoding="UTF8") as fx, open(
            y_dev_path, "r", encoding="UTF8"
        ) as fy:
            for data, golds in zip(fx, fy):
                data, golds = data.strip(), golds.strip().split("\t")
                trial_pairs[data] = []
                for gold in golds:
                    trial_pairs[data].append(gold)

        trial_queries = list(trial_pairs.keys())

        # model train
        if_freeze_bert = False
        best_MAP = -1
        start_time = time.time()

        for i in range(num_of_epochs):
            print("Epoch: #{}".format(i + 1))

            # freeze BERT
            if freeze_bert:
                if i < 5:
                    if_freeze_bert = False
                    print("Bert is not freezed")
                else:
                    if_freeze_bert = True
                    print("Bert is freezed")
                if if_freeze_bert:
                    model.module.freeze_bert()

            steps = 0
            epoch_loss = 0
            size = len(train_dataset) // batch_size
            if len(train_dataset) % batch_size:
                size += 1

            for batch in tqdm(
                generate_batch(
                    train_dataset,
                    gold_dict,
                    vocab_sampler,
                    subsample,
                    pos_sample_prob,
                    batch_size,
                    n_negs,
                ),
                total=size,
            ):
                encodings, query_mask, hyper_mask, labels = batch
                input_ids = encodings["input_ids"].cuda()
                attention_mask = encodings["attention_mask"].cuda()
                query_mask = query_mask.cuda()
                hyper_mask = hyper_mask.cuda()
                labels = labels.cuda()

                outputs = torch.flatten(model(input_ids, attention_mask, query_mask, hyper_mask))
                optimizer.zero_grad()
                loss = loss_fn(outputs, labels.float())
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                steps += 1

                del input_ids, attention_mask, query_mask, hyper_mask, labels

            epoch_loss = epoch_loss / steps

            print(f"Training Finised")
            print(f"Epoch [{i+1}/{num_of_epochs}] Loss : {epoch_loss:.6f} ")
            MAP_ = evaluator.get_MAP(trial_queries, trial_pairs)
            print(f"Evaluation MAP: {MAP_:.6f}")

            if MAP_ > best_MAP:
                best_MAP = MAP_
                torch.save(model, model_save_path)

            check_point_data = []
            check_point_data.append(str(i + 1))
            check_point_data.append(f"{epoch_loss:.4f}")
            check_point_data.append(f"{MAP_:.4f}")
            check_point_data.append(f"{time.time()-start_time:.2f}s")

            with open(log_path, "a") as fa:
                fa.write("\t".join(check_point_data) + "\n")

        # infer immediately
        del model
        model = torch.load(model_save_path)
        model.cuda()

        result_save_dir = f"./outputs/results/{task}_model/"
        if subsample:
            result_save_dir += "_subsample"
        if freeze_bert:
            result_save_dir += "_freeze"

        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)
        dataset_dir = f"./dataset/shuffled_dataset/data_{str(n).zfill(2)}"
        x_test_path = os.path.join(dataset_dir, f"data/{task_dict[task]}.test.data.txt")
        y_test_path = os.path.join(dataset_dir, f"gold/{task_dict[task]}.test.gold.txt")
        save_path = os.path.join(result_save_dir, f"{str(n).zfill(2)}_output.txt")
        test_pairs = {}
        with open(x_test_path, "r", encoding="UTF8") as fx, open(
            y_test_path, "r", encoding="UTF8"
        ) as fy:
            for data, golds in zip(fx, fy):
                data, golds = data.strip(), golds.strip().split("\t")

                test_pairs[data] = []
                for gold in golds:
                    test_pairs[data].append(gold)

        test_queries = list(test_pairs.keys())
        predictor = HypertPredictor(model, tokenizer, 32768, task, 15)
        hypernyms = predictor.pred_hypernyms(test_queries)
        save_result(hypernyms, save_path)

        del model, predictor


if __name__ == "__main__":
    main()
