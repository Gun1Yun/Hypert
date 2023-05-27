import sys
import os
import argparse
import numpy as np

from task9_scorer import *

LIMIT = 15
TASK_DICT = {"1A": "1A.english", "2A": "2A.medical", "2B": "2B.music"}


def get_score(golds_path, preds_path):
    all_scores = []
    fgold = open(golds_path, "r")
    fpredictions = open(preds_path, "r")

    golds = fgold.readlines()
    preds = fpredictions.readlines()

    for i in range(len(golds)):
        gold_line = golds[i]
        pred_line = preds[i]

        avg_pat1 = []
        avg_pat2 = []
        avg_pat3 = []
        avg_pat4 = []

        gold_hypernyms = get_hypernyms(gold_line, is_gold=True)
        pred_hypernyms = get_hypernyms(pred_line, is_gold=False)
        n_golds = len(gold_hypernyms)
        r = [0 for _ in range(LIMIT)]

        for i in range(len(pred_hypernyms)):
            pred_hyp = pred_hypernyms[i]
            if pred_hyp in gold_hypernyms:
                r[i] = 1

        avg_pat1.append(precision_at_k(r, 1, n_golds))
        avg_pat2.append(precision_at_k(r, 3, n_golds))
        avg_pat3.append(precision_at_k(r, 5, n_golds))
        avg_pat4.append(precision_at_k(r, 15, n_golds))

        mrr_score_numb = mean_reciprocal_rank(r)
        map_score_numb = mean_average_precision(r, n_golds)
        avg_pat1_numb = sum(avg_pat1) / len(avg_pat1)
        avg_pat2_numb = sum(avg_pat2) / len(avg_pat2)
        avg_pat3_numb = sum(avg_pat3) / len(avg_pat3)
        avg_pat4_numb = sum(avg_pat4) / len(avg_pat4)

        scores_results = [
            mrr_score_numb,
            map_score_numb,
            avg_pat1_numb,
            avg_pat2_numb,
            avg_pat3_numb,
            avg_pat4_numb,
        ]
        all_scores.append(scores_results)

    result_scores = []
    for k in range(len(scores_names)):
        result_scores.append(
            str(sum([score_list[k] for score_list in all_scores]) / len(all_scores))
        )

    return result_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="Task 1A | 2A | 2B")

    args = parser.parse_args()

    task = args.task

    save_path = ""

    scores_names = ["MRR", "MAP", "P@1", "P@3", "P@5", "P@15"]

    n_repeats = 10
    final_scores = []
    for i in range(1, n_repeats + 1):
        dataset_dir = f"./dataset/shuffled_dataset/data_{str(i).zfill(2)}"
        gold_path = os.path.join(dataset_dir, f"gold/{TASK_DICT[task]}.test.gold.txt")
        pred_path = f"./outputs/results/{task}_model/{str(i).zfill(2)}_output.txt"

        if not os.path.exists(pred_path):
            continue
        
        scores = get_score(gold_path, pred_path)
        scores = np.array(scores, dtype=np.float32)
        final_scores.append(scores)

    final_scores = np.array(final_scores)
    mean_scores = final_scores.mean(axis=0) * 100
    std_scores = final_scores.std(axis=0) * 100

    for idx, score_name in enumerate(scores_names):
        print(f"{score_name}: {mean_scores[idx]:.2f} +- {std_scores[idx]:.2f}")
