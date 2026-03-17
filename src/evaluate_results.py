import argparse
import ast
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import f1_score

from load_data import load_dataset


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

# VizWiz normalization and variable naming are intentionally aligned with the
# official evaluation API style (vqaEval.py) for consistency.
# Reference: https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/API.zip

punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

manualMap = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}

articles = ["a", "an", "the"]

periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile(r"(\d)(\,)(\d)")

contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", "couldn'tve": "couldn’t’ve", "couldnt’ve": "couldn’t’ve", "didnt": "didn’t", "doesnt": "doesn’t", "dont": "don’t", "hadnt": "hadn’t", "hadnt’ve": "hadn’t’ve", "hadn'tve": "hadn’t’ve", "hasnt": "hasn’t", "havent": "haven’t", "hed": "he’d", "hed’ve": "he’d’ve", "he’dve": "he’d’ve", "hes": "he’s", "howd": "how’d", "howll": "how’ll", "hows": "how’s", "Id’ve": "I’d’ve", "I’dve": "I’d’ve", "Im": "I’m", "Ive": "I’ve", "isnt": "isn’t", "itd": "it’d", "itd’ve": "it’d’ve", "it’dve": "it’d’ve", "itll": "it’ll", "let’s": "let’s", "maam": "ma’am", "mightnt": "mightn’t", "mightnt’ve": "mightn’t’ve", "mightn’tve": "mightn’t’ve", "mightve": "might’ve", "mustnt": "mustn’t", "mustve": "must’ve", "neednt": "needn’t", "notve": "not’ve", "oclock": "o’clock", "oughtnt": "oughtn’t", "ow’s’at": "’ow’s’at", "’ows’at": "’ow’s’at", "’ow’sat": "’ow’s’at", "shant": "shan’t", "shed’ve": "she’d’ve", "she’dve": "she’d’ve", "she’s": "she’s", "shouldve": "should’ve", "shouldnt": "shouldn’t", "shouldnt’ve": "shouldn’t’ve", "shouldn’tve": "shouldn’t’ve", "somebody’d": "somebodyd", "somebodyd’ve": "somebody’d’ve", "somebody’dve": "somebody’d’ve", "somebodyll": "somebody’ll", "somebodys": "somebody’s", "someoned": "someone’d", "someoned’ve": "someone’d’ve", "someone’dve": "someone’d’ve", "someonell": "someone’ll", "someones": "someone’s", "somethingd": "something’d", "somethingd’ve": "something’d’ve", "something’dve": "something’d’ve", "somethingll": "something’ll", "thats": "that’s", "thered": "there’d", "thered’ve": "there’d’ve", "there’dve": "there’d’ve", "therere": "there’re", "theres": "there’s", "theyd": "they’d", "theyd’ve": "they’d’ve", "they’dve": "they’d’ve", "theyll": "they’ll", "theyre": "they’re", "theyve": "they’ve", "twas": "’twas", "wasnt": "wasn’t", "wed’ve": "we’d’ve", "we’dve": "we’d’ve", "weve": "we've", "werent": "weren’t", "whatll": "what’ll", "whatre": "what’re", "whats": "what’s", "whatve": "what’ve", "whens": "when’s", "whered": "where’d", "wheres": "where's", "whereve": "where’ve", "whod": "who’d", "whod’ve": "who’d’ve", "who’dve": "who’d’ve", "wholl": "who’ll", "whos": "who’s", "whove": "who've", "whyll": "why’ll", "whyre": "why’re", "whys": "why’s", "wont": "won’t", "wouldve": "would’ve", "wouldnt": "wouldn’t", "wouldnt’ve": "wouldn’t’ve", "wouldn’tve": "wouldn’t’ve", "yall": "y’all", "yall’ll": "y’all’ll", "y’allll": "y’all’ll", "yall’d’ve": "y’all’d’ve", "y’alld’ve": "y’all’d’ve", "y’all’dve": "y’all’d’ve", "youd": "you’d", "youd’ve": "you’d’ve", "you’dve": "you’d’ve", "youll": "you’ll", "youre": "you’re", "youve": "you’ve"}

def processPunctuation(inText: str) -> str:
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText: str) -> str:
    words = []
    for word in inText.lower().split():
        word = manualMap.get(word, word)
        if word in articles:
            continue
        words.append(contractions.get(word, word))
    return " ".join(words)


def _compute_vizwiz_scores(items: List[Dict]) -> Tuple[float, float, Dict[str, Tuple[float, float]], int]:
    anno_file = REPO_ROOT / "data" / "vizwiz" / "val.json"
    if not anno_file.exists():
        raise FileNotFoundError(f"VizWiz annotation file not found: {anno_file}")

    with open(anno_file, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    id2answers: Dict[str, List[Dict]] = {}
    id2type: Dict[str, str] = {}
    for ann in annotations:
        key = str(ann.get("image", ""))
        if not key:
            continue
        id2answers[key] = ann.get("answers", [])
        id2type[key] = ann.get("answer_type", "unknown")

    accQA: List[float] = []
    f1QA: List[float] = []
    type2acc: Dict[str, List[float]] = {}
    type2f1: Dict[str, List[float]] = {}
    skipped = 0

    for item in items:
        sid = str(item.get("id", ""))
        if sid not in id2answers:
            skipped += 1
            continue

        resAns = str(item.get("pred_answer", "")).replace("\n", " ").replace("\t", " ").strip()
        resAns = processDigitArticle(processPunctuation(resAns))

        correctness: List[float] = []
        f1: List[float] = []
        for ans in id2answers[sid]:
            if ans.get("answer_confidence") == "no":
                continue
            gt_ans = str(ans.get("answer", "")).replace("\n", " ").replace("\t", " ").strip()
            gt_ans = processDigitArticle(processPunctuation(gt_ans))
            correctness.append(1.0 if gt_ans == resAns else 0.0)
            f1.append(_compute_token_f1(resAns, gt_ans))

        if not correctness:
            skipped += 1
            continue

        best_acc = max(correctness)
        best_f1 = max(f1)
        accQA.append(best_acc)
        f1QA.append(best_f1)

        qtype = id2type.get(sid, "unknown")
        type2acc.setdefault(qtype, []).append(best_acc)
        type2f1.setdefault(qtype, []).append(best_f1)

    if not accQA:
        raise ValueError("No VizWiz records could be scored.")

    per_type: Dict[str, Tuple[float, float]] = {}
    for qtype in sorted(type2acc):
        per_type[qtype] = (
            sum(type2acc[qtype]) / len(type2acc[qtype]),
            sum(type2f1[qtype]) / len(type2f1[qtype]),
        )

    return sum(accQA) / len(accQA), sum(f1QA) / len(f1QA), per_type, skipped


def _compute_token_f1(pred: str, gt: str) -> float:
    translator = str.maketrans("", "", string.punctuation)
    pred_tokens = pred.strip().lower().translate(translator).split()
    gt_tokens = gt.strip().lower().translate(translator).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _compute_weighted_f1(items: List[Dict]) -> float:
    labels = sorted({it["gt_answer"].strip().lower() for it in items})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_true = [label_to_idx[it["gt_answer"].strip().lower()] for it in items]
    y_pred = [label_to_idx[it["pred_answer"].strip().lower()] if it["pred_answer"].strip().lower() in label_to_idx else -1 for it in items]
    return f1_score(y_true, y_pred, average="weighted", labels=list(label_to_idx.values()))


def _load_items(path: str) -> Tuple[List[Dict], int, int]:
    items: List[Dict] = []
    total_lines = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                item = ast.literal_eval(line)
            items.append(item)

    dedup: List[Dict] = []
    seen = set()
    for item in items:
        sid = str(item.get("id", ""))
        if sid and sid in seen:
            continue
        seen.add(sid)
        dedup.append(item)
    duplicate_count = max(0, len(items) - len(dedup))
    return dedup, total_lines, duplicate_count


def _build_completeness_message(pred_count: int, expected_count: int) -> str:
    if expected_count <= 0:
        return f"Prediction count: {pred_count} (expected count unavailable)"
    if pred_count >= expected_count:
        return f"Prediction coverage: COMPLETE ({pred_count}/{expected_count})"
    missing = expected_count - pred_count
    return f"Prediction coverage: INCOMPLETE ({pred_count}/{expected_count}, missing {missing})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction jsonl")
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, choices=["okvqa", "vizwiz", "cub", "flowers"])
    args = parser.parse_args()

    if not os.path.exists(args.pred_file):
        raise FileNotFoundError(f"Prediction file not found: {args.pred_file}")

    items, total_lines, duplicate_count = _load_items(args.pred_file)
    if not items:
        raise ValueError("No valid prediction records found.")

    expected_count = -1
    try:
        test_dataset = load_dataset(args.data, split="test")
        expected_count = len(test_dataset)
    except Exception as exc:
        print(f"Warning: could not load test dataset to verify completeness ({exc}).")

    print(_build_completeness_message(pred_count=len(items), expected_count=expected_count))
    if duplicate_count > 0:
        print(f"Note: removed {duplicate_count} duplicate prediction records by id.")
    if total_lines != len(items):
        print(f"Evaluating {len(items)} unique records from {total_lines} file lines.")

    if args.data in {"cub", "flowers"}:
        acc = sum(it["pred_answer"].strip().lower() == it["gt_answer"].strip().lower() for it in items) / len(items)
        wf1 = _compute_weighted_f1(items)
        print(f"Accuracy: {acc * 100:.2f}% | Weighted F1: {wf1 * 100:.2f}%")
        return

    if args.data == "okvqa":
        em = sum(it["pred_answer"].strip().lower() == it["gt_answer"].strip().lower() for it in items) / len(items)
        f1 = sum(_compute_token_f1(it["pred_answer"], it["gt_answer"]) for it in items) / len(items)
        print(f"EM: {em * 100:.2f}% | F1: {f1 * 100:.2f}%")
        return

    viz_em, viz_f1, per_type_scores, skipped = _compute_vizwiz_scores(items)
    print(f"EM: {viz_em * 100:.2f}% | F1: {viz_f1 * 100:.2f}%")
    for qtype in sorted(per_type_scores):
        t_em, t_f1 = per_type_scores[qtype]
        print(f"[{qtype} questions] EM: {t_em * 100:.2f}% | F1: {t_f1 * 100:.2f}%")
    if skipped > 0:
        print(f"Note: skipped {skipped} records that could not be matched/scored by VizWiz annotations.")


if __name__ == "__main__":
    main()
