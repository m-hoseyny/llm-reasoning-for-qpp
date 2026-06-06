#!/usr/bin/env python3
"""Evaluate LLM reason labels against human annotations.

The annotation CSVs are expected to contain adjacent pairs of rows per query:
one row with annotation_type == "LLM judgement" and one row with
annotation_type == "Human annotation".
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


METADATA_COLUMNS = {"source_split", "qid", "annotation_type", "query"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LLM-vs-human metrics for DRAQ reason annotations."
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the annotation CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="DRAQ Human Annotation - *.csv",
        help="Glob pattern for annotation files inside --annotation-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metric CSV outputs. Defaults to --annotation-dir/metrics.",
    )
    parser.add_argument(
        "--llm-source",
        choices=("first", "per-file"),
        default="first",
        help=(
            "Use LLM labels from the first annotation file for all comparisons, "
            "or use each file's own LLM row."
        ),
    )
    return parser.parse_args()


def read_annotation_file(path: Path) -> tuple[list[str], dict[str, dict[str, object]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} is empty or missing a header")

        required = {"source_split", "qid", "annotation_type", "query"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        reason_columns = [c for c in reader.fieldnames if c not in METADATA_COLUMNS]
        rows = list(reader)

    if len(rows) % 2 != 0:
        raise ValueError(f"{path} has {len(rows)} data rows; expected an even number")

    examples: dict[str, dict[str, object]] = {}
    for index in range(0, len(rows), 2):
        llm_row = rows[index]
        human_row = rows[index + 1]

        if llm_row["annotation_type"].strip() != "LLM judgement":
            raise ValueError(
                f"{path}:{index + 2} expected annotation_type='LLM judgement'"
            )
        if human_row["annotation_type"].strip() != "Human annotation":
            raise ValueError(
                f"{path}:{index + 3} expected annotation_type='Human annotation'"
            )

        key = make_key(llm_row)
        human_key = make_key(human_row)
        if key != human_key:
            raise ValueError(f"{path}:{index + 2}-{index + 3} has mismatched pair keys")
        if key in examples:
            raise ValueError(f"{path} has duplicate query key: {key}")

        examples[key] = {
            "source_split": llm_row["source_split"].strip(),
            "qid": llm_row["qid"].strip(),
            "query": llm_row["query"].strip(),
            "llm": parse_labels(llm_row, reason_columns, path, index + 2),
            "human": parse_labels(human_row, reason_columns, path, index + 3),
        }

    return reason_columns, examples


def make_key(row: dict[str, str]) -> str:
    return f"{row['source_split'].strip()}::{row['qid'].strip()}::{row['query'].strip()}"


def parse_labels(
    row: dict[str, str], reason_columns: list[str], path: Path, line_number: int
) -> list[int]:
    labels: list[int] = []
    for column in reason_columns:
        value = row[column].strip()
        if value not in {"0", "1"}:
            raise ValueError(
                f"{path}:{line_number} column '{column}' must be 0 or 1, got {value!r}"
            )
        labels.append(int(value))
    return labels


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)
    balanced_accuracy = (recall + specificity) / 2
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate_predictions(
    name: str,
    keys: list[str],
    reason_columns: list[str],
    llm_by_key: dict[str, list[int]],
    human_by_key: dict[str, list[int]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    y_true = list(itertools.chain.from_iterable(human_by_key[key] for key in keys))
    y_pred = list(itertools.chain.from_iterable(llm_by_key[key] for key in keys))
    micro = binary_metrics(y_true, y_pred)

    per_reason: list[dict[str, object]] = []
    for idx, reason in enumerate(reason_columns):
        reason_true = [human_by_key[key][idx] for key in keys]
        reason_pred = [llm_by_key[key][idx] for key in keys]
        metrics = binary_metrics(reason_true, reason_pred)
        per_reason.append({"comparison": name, "reason": reason, **metrics})

    macro_precision = mean(float(row["precision"]) for row in per_reason)
    macro_recall = mean(float(row["recall"]) for row in per_reason)
    macro_f1 = mean(float(row["f1"]) for row in per_reason)
    macro_balanced_accuracy = mean(float(row["balanced_accuracy"]) for row in per_reason)

    exact_matches = 0
    jaccards: list[float] = []
    hamming_losses: list[float] = []
    per_query: list[dict[str, object]] = []
    for key in keys:
        human = human_by_key[key]
        llm = llm_by_key[key]
        exact_match = int(human == llm)
        exact_matches += exact_match

        intersection = sum(1 for h, l in zip(human, llm) if h == 1 and l == 1)
        union = sum(1 for h, l in zip(human, llm) if h == 1 or l == 1)
        jaccard = 1.0 if union == 0 else intersection / union
        hamming_loss = sum(1 for h, l in zip(human, llm) if h != l) / len(reason_columns)
        jaccards.append(jaccard)
        hamming_losses.append(hamming_loss)
        split, qid, query = key.split("::", 2)
        per_query.append(
            {
                "comparison": name,
                "source_split": split,
                "qid": qid,
                "query": query,
                "exact_match": exact_match,
                "jaccard": jaccard,
                "hamming_loss": hamming_loss,
                "llm_positive_count": sum(llm),
                "human_positive_count": sum(human),
            }
        )

    summary = {
        "comparison": name,
        "num_queries": len(keys),
        "num_reason_decisions": len(y_true),
        "label_accuracy": micro["accuracy"],
        "exact_match_accuracy": safe_divide(exact_matches, len(keys)),
        "mean_jaccard": mean(jaccards),
        "mean_hamming_loss": mean(hamming_losses),
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "micro_specificity": micro["specificity"],
        "micro_balanced_accuracy": micro["balanced_accuracy"],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_balanced_accuracy": macro_balanced_accuracy,
        "tp": micro["tp"],
        "fp": micro["fp"],
        "fn": micro["fn"],
        "tn": micro["tn"],
        "llm_positive_rate": safe_divide(sum(y_pred), len(y_pred)),
        "human_positive_rate": safe_divide(sum(y_true), len(y_true)),
    }
    return summary, per_reason, per_query


def evaluate_pooled_humans(
    name: str,
    keys: list[str],
    reason_columns: list[str],
    llm_by_key: dict[str, list[int]],
    human_labels_by_file: list[dict[str, list[int]]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    exact_matches = 0
    jaccards: list[float] = []
    hamming_losses: list[float] = []

    for human_by_key in human_labels_by_file:
        for key in keys:
            human = human_by_key[key]
            llm = llm_by_key[key]
            y_true.extend(human)
            y_pred.extend(llm)

            exact_matches += int(human == llm)
            intersection = sum(1 for h, l in zip(human, llm) if h == 1 and l == 1)
            union = sum(1 for h, l in zip(human, llm) if h == 1 or l == 1)
            jaccards.append(1.0 if union == 0 else intersection / union)
            hamming_losses.append(
                sum(1 for h, l in zip(human, llm) if h != l) / len(reason_columns)
            )

    micro = binary_metrics(y_true, y_pred)

    per_reason: list[dict[str, object]] = []
    for idx, reason in enumerate(reason_columns):
        reason_true: list[int] = []
        reason_pred: list[int] = []
        for human_by_key in human_labels_by_file:
            for key in keys:
                reason_true.append(human_by_key[key][idx])
                reason_pred.append(llm_by_key[key][idx])
        metrics = binary_metrics(reason_true, reason_pred)
        per_reason.append({"comparison": name, "reason": reason, **metrics})

    total_query_annotations = len(keys) * len(human_labels_by_file)
    summary = {
        "comparison": name,
        "num_queries": len(keys),
        "num_reason_decisions": len(y_true),
        "label_accuracy": micro["accuracy"],
        "exact_match_accuracy": safe_divide(exact_matches, total_query_annotations),
        "mean_jaccard": mean(jaccards),
        "mean_hamming_loss": mean(hamming_losses),
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "micro_specificity": micro["specificity"],
        "micro_balanced_accuracy": micro["balanced_accuracy"],
        "macro_precision": mean(float(row["precision"]) for row in per_reason),
        "macro_recall": mean(float(row["recall"]) for row in per_reason),
        "macro_f1": mean(float(row["f1"]) for row in per_reason),
        "macro_balanced_accuracy": mean(
            float(row["balanced_accuracy"]) for row in per_reason
        ),
        "tp": micro["tp"],
        "fp": micro["fp"],
        "fn": micro["fn"],
        "tn": micro["tn"],
        "llm_positive_rate": safe_divide(sum(y_pred), len(y_pred)),
        "human_positive_rate": safe_divide(sum(y_true), len(y_true)),
    }
    return summary, per_reason


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def majority_vote(labels: list[list[int]]) -> list[int]:
    return [int(sum(values) >= math.ceil(len(values) / 2)) for values in zip(*labels)]


def cohen_kappa(a: list[int], b: list[int]) -> float:
    if len(a) != len(b):
        raise ValueError("Cohen kappa inputs must have the same length")
    n = len(a)
    observed = safe_divide(sum(1 for x, y in zip(a, b) if x == y), n)
    counts_a = Counter(a)
    counts_b = Counter(b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in (0, 1))
    return safe_divide(observed - expected, 1 - expected)


def fleiss_kappa(human_labels_by_file: list[dict[str, list[int]]], keys: list[str]) -> float:
    if not human_labels_by_file:
        return 0.0

    n_annotators = len(human_labels_by_file)
    items: list[list[int]] = []
    for key in keys:
        for reason_idx in range(len(next(iter(human_labels_by_file[0].values())))):
            items.append([labels[key][reason_idx] for labels in human_labels_by_file])

    if n_annotators < 2 or not items:
        return 0.0

    p_items = []
    total_label_counts = Counter()
    for item in items:
        counts = Counter(item)
        total_label_counts.update(item)
        agreement = sum(count * count for count in counts.values()) - n_annotators
        p_items.append(agreement / (n_annotators * (n_annotators - 1)))

    p_bar = mean(p_items)
    total_assignments = len(items) * n_annotators
    p_e = sum((total_label_counts[label] / total_assignments) ** 2 for label in (0, 1))
    return safe_divide(p_bar - p_e, 1 - p_e)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.annotation_dir / "metrics"
    paths = sorted(args.annotation_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(
            f"No annotation files found in {args.annotation_dir} with pattern {args.pattern!r}"
        )

    all_examples: dict[str, dict[str, dict[str, object]]] = {}
    reason_columns: list[str] | None = None
    for path in paths:
        file_reason_columns, examples = read_annotation_file(path)
        if reason_columns is None:
            reason_columns = file_reason_columns
        elif reason_columns != file_reason_columns:
            raise ValueError(f"{path} has different reason columns than the first file")
        all_examples[path.name] = examples

    assert reason_columns is not None

    key_sets = [set(examples) for examples in all_examples.values()]
    shared_keys = sorted(set.intersection(*key_sets))
    if not shared_keys:
        raise ValueError("Annotation files do not share any query keys")

    warnings: list[str] = []
    for path_name, examples in all_examples.items():
        missing = set(shared_keys) - set(examples)
        if missing:
            warnings.append(f"{path_name} is missing {len(missing)} shared keys")

    first_file = paths[0].name
    canonical_llm = {
        key: all_examples[first_file][key]["llm"] for key in shared_keys
    }
    for key in shared_keys:
        variants = {
            path_name: tuple(examples[key]["llm"])
            for path_name, examples in all_examples.items()
            if key in examples
        }
        if len(set(variants.values())) > 1:
            warnings.append(
                f"LLM label mismatch for {key}; defaulting to {first_file} labels"
            )

    summary_rows: list[dict[str, object]] = []
    per_reason_rows: list[dict[str, object]] = []
    per_query_rows: list[dict[str, object]] = []

    human_labels_by_file: list[dict[str, list[int]]] = []
    for path_name, examples in all_examples.items():
        human_by_key = {key: examples[key]["human"] for key in shared_keys}
        human_labels_by_file.append(human_by_key)
        llm_by_key = (
            {key: examples[key]["llm"] for key in shared_keys}
            if args.llm_source == "per-file"
            else canonical_llm
        )
        summary, per_reason, per_query = evaluate_predictions(
            path_name, shared_keys, reason_columns, llm_by_key, human_by_key
        )
        summary_rows.append(summary)
        per_reason_rows.extend(per_reason)
        per_query_rows.extend(per_query)

    summary, per_reason = evaluate_pooled_humans(
        "all_human_annotations",
        shared_keys,
        reason_columns,
        canonical_llm,
        human_labels_by_file,
    )
    summary_rows.append(summary)
    per_reason_rows.extend(per_reason)

    majority_human = {
        key: majority_vote([labels[key] for labels in human_labels_by_file])
        for key in shared_keys
    }
    summary, per_reason, per_query = evaluate_predictions(
        "majority_vote_human",
        shared_keys,
        reason_columns,
        canonical_llm,
        majority_human,
    )
    summary_rows.append(summary)
    per_reason_rows.extend(per_reason)
    per_query_rows.extend(per_query)

    agreement_rows: list[dict[str, object]] = []
    for (file_a, examples_a), (file_b, examples_b) in itertools.combinations(
        all_examples.items(), 2
    ):
        labels_a = list(
            itertools.chain.from_iterable(examples_a[key]["human"] for key in shared_keys)
        )
        labels_b = list(
            itertools.chain.from_iterable(examples_b[key]["human"] for key in shared_keys)
        )
        agreement_rows.append(
            {
                "annotator_a": file_a,
                "annotator_b": file_b,
                "num_reason_decisions": len(labels_a),
                "percent_agreement": safe_divide(
                    sum(1 for a, b in zip(labels_a, labels_b) if a == b), len(labels_a)
                ),
                "cohen_kappa": cohen_kappa(labels_a, labels_b),
            }
        )
    agreement_rows.append(
        {
            "annotator_a": "all_humans",
            "annotator_b": "all_humans",
            "num_reason_decisions": len(shared_keys) * len(reason_columns),
            "percent_agreement": "",
            "cohen_kappa": fleiss_kappa(human_labels_by_file, shared_keys),
        }
    )

    write_csv(output_dir / "llm_vs_human_summary.csv", summary_rows)
    write_csv(output_dir / "llm_vs_human_per_reason.csv", per_reason_rows)
    write_csv(output_dir / "llm_vs_human_per_query.csv", per_query_rows)
    write_csv(output_dir / "human_inter_annotator_agreement.csv", agreement_rows)

    print(f"Annotation files: {len(paths)}")
    print(f"Shared queries: {len(shared_keys)}")
    print(f"Reason columns: {len(reason_columns)}")
    print(f"LLM source: {args.llm_source}")
    print()
    print("LLM vs human summary")
    header = [
        "comparison",
        "label_accuracy",
        "exact_match_accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_f1",
        "mean_jaccard",
    ]
    print(",".join(header))
    for row in summary_rows:
        print(",".join(format_float(row[column]) for column in header))

    print()
    print("Human inter-annotator agreement")
    for row in agreement_rows:
        print(
            f"{row['annotator_a']} vs {row['annotator_b']}: "
            f"agreement={format_float(row['percent_agreement'])}, "
            f"kappa={format_float(row['cohen_kappa'])}"
        )

    if warnings:
        warning_path = output_dir / "warnings.txt"
        warning_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        print()
        print(f"Warnings: {len(warnings)} written to {warning_path}")

    print()
    print(f"Wrote metrics to: {output_dir}")


if __name__ == "__main__":
    main()
