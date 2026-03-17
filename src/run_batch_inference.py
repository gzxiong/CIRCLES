import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, Set

from tqdm import tqdm
from vllm import LLM, SamplingParams

from load_data import load_dataset
from visual_icl import VisualICL


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent


def _load_processed_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()

    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                item = ast.literal_eval(line)
            sid = str(item.get("id", ""))
            if sid:
                ids.add(sid)
    return ids


def _build_save_path(
    save_root: str,
    data: str,
    model_name: str,
    method: str,
    k: int,
    num_attributes: int,
    attribute_k: int,
) -> str:
    model_dir = model_name.replace("/", "_")
    save_dir = os.path.join(save_root, data, model_dir)
    os.makedirs(save_dir, exist_ok=True)
    if method in {"none", "random", "rices", "muier", "mmices"}:
        fname = f"{method}_{k}.jsonl"
    else:
        fname = f"{method}_{k}_{num_attributes}_{attribute_k}.jsonl"
    return os.path.join(save_dir, fname)


def _to_record(item: Dict, output: Dict) -> Dict:
    return {
        "id": item["id"],
        "question": item["question"],
        "imgpath": item["imgpath"],
        "pred_answer": output.get("answer", ""),
        "gt_answer": item.get("answer", ""),
        "method": output.get("method"),
        "k": output.get("k"),
        "num_attributes": output.get("num_attributes"),
        "attribute_k": output.get("attribute_k"),
        "retrieved_examples": output.get("retrieved_examples"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference with UnifiedVisualICL")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        choices=[
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
            "OpenGVLab/InternVL3_5-4B-Instruct",
            "OpenGVLab/InternVL3_5-8B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "google/gemma-3-27b-it",
        ],
    )
    parser.add_argument("--data", type=str, default="okvqa", choices=["okvqa", "vizwiz", "cub", "flowers"])
    parser.add_argument(
        "--method",
        type=str,
        default="none",
        choices=["none", "random", "rices", "muier", "mmices", "circles"],
    )
    parser.add_argument("--clip_model", type=str, default="laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--num_attributes", type=int, default=1)
    parser.add_argument("--attribute_k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--limit_mm_per_prompt", type=int, default=36)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--save_root", type=str, default=str(REPO_ROOT / "results"))
    parser.add_argument("--n", type=int, default=1, help="Number of shards")
    parser.add_argument("--i", type=int, default=0, help="Current shard index")
    args = parser.parse_args()

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    vlm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt, "video": 0},
        mm_processor_cache_gb=0,
        trust_remote_code=True,
    )

    train_dataset = load_dataset(args.data, split="train")
    test_dataset = load_dataset(args.data, split="test")

    icl = VisualICL(
        vlm=vlm,
        method=args.method,
        clip_model_name=args.clip_model,
        sampling_params=sampling_params,
        use_tqdm=False,
        default_k=args.k,
        num_attributes=args.num_attributes,
        attribute_k=args.attribute_k,
    )

    save_path = _build_save_path(
        save_root=args.save_root,
        data=args.data,
        model_name=args.model,
        method=args.method,
        k=args.k,
        num_attributes=args.num_attributes,
        attribute_k=args.attribute_k,
    )
    processed_ids = _load_processed_ids(save_path)

    indices = [idx for idx in range(len(test_dataset)) if idx % args.n == args.i]
    for idx in tqdm(indices, desc="Running inference"):
        item = test_dataset[idx]
        if item["id"] in processed_ids:
            continue
        output = icl.predict(
            question=item["question"],
            query_image=item["imgpath"],
            train_dataset=train_dataset,
            k=args.k,
            task=test_dataset.task,
            options=getattr(test_dataset, "options", None),
            num_attributes=args.num_attributes,
            attribute_k=args.attribute_k,
        )
        record = _to_record(item, output)
        with open(save_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
