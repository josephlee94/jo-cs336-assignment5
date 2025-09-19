# scripts/eval_math_baseline.py
import argparse, json, re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_jgrader import r1_zero_reward_fn  # ← official reward fn


# -----------------------
# JSON / JSONL loader
# -----------------------
def _load_json_or_jsonl(path: Path) -> List[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# -----------------------
# r1_zero prompt
# -----------------------
R1_ZERO_PATH = Path("cs336_alignment/prompts/r1_zero.prompt")

def load_r1_zero_template() -> str:
    with open(R1_ZERO_PATH, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------
# Format check / extraction (for format_reward only)
# -----------------------
ANSWER_RE = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.IGNORECASE | re.DOTALL)

def has_answer_block(text: Optional[str]) -> bool:
    return ANSWER_RE.search(text or "") is not None


# -----------------------
# Dataset loader
# -----------------------
def load_math_like(path: Path, template: str) -> List[Tuple[str, str]]:
    """
    Accepts either:
      • MATH-style: [{"problem": "...", "answer": "..."}]
      • GSM8K-style: [{"question": "...", "answer": "#### 42"}, ...]
    Returns list of (prompt_text, ground_truth_answer).
    """
    data = _load_json_or_jsonl(path)
    items: List[Tuple[str, str]] = []
    for ex in data:
        if "problem" in ex and "answer" in ex:  # MATH style
            q = ex["problem"]
            a = ex["answer"]
        elif "question" in ex and "answer" in ex:  # GSM8K style
            q = ex["question"]
            a = ex["answer"]
            m = re.search(r"####\s*(.+)", a)
            if m:
                a = m.group(1).strip()
        else:
            q = ex.get("question", ex.get("prompt", ex.get("input", "")))
            a = ex.get("answer", ex.get("output", ex.get("target", "")))

        prompt = template.format(question=q)
        items.append((prompt, a))
    return items


# -----------------------
# vLLM evaluation
# -----------------------
def evaluate_vllm(
    vllm_model: str,
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    max_batch_size: int = 32,
) -> List[str]:
    llm = LLM(model=vllm_model, trust_remote_code=True)
    generations: List[str] = []

    for i in range(0, len(prompts), max_batch_size):
        batch = prompts[i : i + max_batch_size]
        outs = llm.generate(batch, eval_sampling_params)
        for out in outs:
            if len(out.outputs) == 0:
                generations.append("")
            else:
                generations.append(out.outputs[0].text)
    return generations


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True,
                    help="JSON or JSONL file with MATH or GSM8K-style fields.")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--temperature", type=float, default=1.0)  # per writeup
    ap.add_argument("--top_p", type=float, default=1.0)        # per writeup
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--out_dir", type=str, default="runs/math_baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_batch_size", type=int, default=16)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load official r1_zero template
    template = load_r1_zero_template()

    # Load dataset & build prompts
    data_pairs = load_math_like(Path(args.dataset_path), template)
    prompts = [p for p, _ in data_pairs]
    gold = [a for _, a in data_pairs]

    # Sampling params — stop when </answer> is produced and KEEP the stop token
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n=1,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Run model
    gens = evaluate_vllm(
        args.model, prompts, sp, max_batch_size=args.max_batch_size
    )

    # Score + serialize
    out_path = Path(args.out_dir) / "results.jsonl"
    fmt1_ans1 = fmt1_ans0 = fmt0_ans1 = fmt0_ans0 = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (prompt, gt, gen) in enumerate(zip(prompts, gold, gens)):
            # format reward: did we see a proper <answer>...</answer>?
            fr = 1 if has_answer_block(gen) else 0
            # answer reward: official r1_zero parser (True/False)
            ar = 1 if r1_zero_reward_fn(gen, gt) else 0

            if fr == 1 and ar == 1:
                fmt1_ans1 += 1
            elif fr == 1 and ar == 0:
                fmt1_ans0 += 1
            elif fr == 0 and ar == 1:
                fmt0_ans1 += 1
            else:
                fmt0_ans0 += 1

            rec: Dict = {
                "id": i,
                "prompt": prompt,
                "generation": gen,
                "ground_truth": gt,
                "format_reward": fr,
                "answer_reward": ar,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    N = len(prompts)
    acc = (fmt1_ans1 + fmt0_ans1) / max(1, N)

    print("\n=== Zero-shot baseline (Qwen2.5-Math-1.5B, r1_zero) ===")
    print(f"Total examples: {N}")
    print(f"(1) format=1, answer=1: {fmt1_ans1}")
    print(f"(2) format=1, answer=0: {fmt1_ans0}")
    print(f"(3) format=0, answer=1: {fmt0_ans1}")
    print(f"(4) format=0, answer=0: {fmt0_ans0}")
    print(f"\nBaseline accuracy (answer_reward): {acc:.4f}")
    print(f"\nSerialized to: {out_path}\n")


if __name__ == "__main__":
    main()
