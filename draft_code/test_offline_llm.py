# scripts/eval_math_baseline.py
import argparse, json
from pathlib import Path
from typing import List

from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn   # ✅ 3. use official reward fn

# ✅ 1. use the r1 zero template
R1_ZERO_PATH = Path("cs336_alignment/prompts/r1_zero.prompt")

def load_r1_zero_template() -> str:
    with open(R1_ZERO_PATH, "r", encoding="utf-8") as f:
        return f.read()

def load_dataset(path: Path) -> List[tuple[str,str]]:
    """ ✅ 2. open GSM8K subset with question/answer keys """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            q = ex["question"]
            # strip the "#### answer" suffix if present
            a = ex["answer"]
            # extract the final number after ####
            import re
            m = re.search(r"####\s*(.+)", a)
            if m:
                a = m.group(1).strip()
            rows.append((q, a))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str,
                    default="data/gsm8k/test_20.jsonl",   # ✅ default to the small GSM8K file
                    help="Path to GSM8K test jsonl subset.")
    ap.add_argument("--model", type=str,
                    default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    args = ap.parse_args()

    template = load_r1_zero_template()
    data = load_dataset(Path(args.dataset_path))
    prompts = [template.format(question=q) for q, _ in data]
    gold = [a for _, a in data]

    # ✅ 4. Generation hyperparameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],                    # stop when </answer> is produced
        include_stop_str_in_output=True
    )

    llm = LLM(model=args.model, trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)

    # Score using the official r1_zero reward
    correct = 0
    for i, out in enumerate(outputs):
        gen = out.outputs[0].text
        scores = r1_zero_reward_fn(gen, gold[i])
        if scores["answer_reward"] > 0.5:
            correct += 1
        print(f"\n=== Example {i} ===")
        print("Prompt:", prompts[i][:80], "...")
        print("Generation:", gen)
        print("Ground truth:", gold[i])
        print("Scores:", scores)

    print("\n=== Summary ===")
    print(f"Total: {len(prompts)} | Correct answers: {correct} | Accuracy: {correct/len(prompts):.4f}")

if __name__ == "__main__":
    main()
