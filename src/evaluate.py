import os
import re
import string
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("../data/results", exist_ok=True)

pilot = pd.read_parquet("../data/processed/pilot.parquet")
classical = pd.read_parquet("../data/processed/classical_results.parquet")
agentic = pd.read_parquet("../data/processed/agentic_results.parquet")

gold = pilot[["id", "answer", "supporting_facts", "num_gold_paragraphs"]].rename(
    columns={"answer": "gold_answer"}
)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_title_recall(pred_titles, gold_titles, num_gold):
    return len(set(pred_titles) & set(gold_titles)) / num_gold


def score(df, method_name):
    df = df.merge(gold, on="id")
    df["em"] = df.apply(lambda r: compute_em(r["answer"], r["gold_answer"]), axis=1)
    df["f1"] = df.apply(lambda r: compute_f1(r["answer"], r["gold_answer"]), axis=1)
    df["title_recall"] = df.apply(
        lambda r: compute_title_recall(
            r["titles"], r["supporting_facts"]["title"], r["num_gold_paragraphs"]
        ),
        axis=1,
    )
    df["total_tokens"] = df["prompt_tokens"] + df["completion_tokens"]
    df["method"] = method_name
    return df


classical_scored = score(classical, "classical")
agentic_scored = score(agentic, "agentic")

cols = [
    "method",
    "type",
    "num_titles_in_question",
    "em",
    "f1",
    "title_recall",
    "latency",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]
combined = pd.concat([classical_scored[cols], agentic_scored[cols]], ignore_index=True)

by_group = combined.groupby(["method", "type", "num_titles_in_question"]).mean(numeric_only=True)
overall = combined.groupby("method").mean(numeric_only=True)

print("=== By type / num_titles_in_question ===")
print(by_group)
print("\n=== Overall, classical vs agentic ===")
print(overall)

combined.to_parquet("../data/results/eval_combined.parquet")
by_group.to_csv("../data/results/eval_summary_by_group.csv")
overall.to_csv("../data/results/eval_summary_overall.csv")
print("\nsaved eval_combined.parquet, eval_summary_by_group.csv, eval_summary_overall.csv to data/results/")

metrics = ["em", "f1", "title_recall", "latency", "total_tokens"]

for metric in metrics:
    fig, ax = plt.subplots(figsize=(6, 4))
    overall[metric].plot(kind="bar", ax=ax, color=["tab:blue", "tab:orange"])
    ax.set_title(f"{metric} — classical vs agentic")
    ax.set_ylabel(metric)
    ax.set_xlabel("method")
    plt.tight_layout()
    fig.savefig(f"../data/results/{metric}_overall.png")
    plt.close(fig)

for metric in metrics:
    pivot = combined.pivot_table(
        index=["type", "num_titles_in_question"], columns="method", values=metric
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"{metric} by type / num_titles_in_question")
    ax.set_ylabel(metric)
    plt.tight_layout()
    fig.savefig(f"../data/results/{metric}_by_group.png")
    plt.close(fig)

print("saved graphs to data/results/")
